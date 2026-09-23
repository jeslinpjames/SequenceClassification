import { NextRequest, NextResponse } from "next/server";
import * as tf from "@tensorflow/tfjs";
import { oneHotEncode, flattenBatch, DatasetRow } from "@/lib/dna";
import { buildModel, HyperParams } from "@/lib/modelBuilder";

export const runtime = "nodejs";
export const maxDuration = 60; // seconds — raise on Vercel Pro if you need longer runs

interface TrainRequestBody {
  rows: DatasetRow[];
  hyperparams: HyperParams;
}

// Simple seeded shuffle so runs are reproducible-ish and val split is stable.
function shuffleIndices(n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

export async function POST(req: NextRequest) {
  let body: TrainRequestBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const { rows, hyperparams } = body;

  if (!Array.isArray(rows) || rows.length < 10) {
    return NextResponse.json(
      { error: "Need at least 10 labeled sequences to train." },
      { status: 400 }
    );
  }

  const positives = rows.filter((r) => r.label === 1).length;
  const negatives = rows.length - positives;
  if (positives === 0 || negatives === 0) {
    return NextResponse.json(
      { error: "Dataset must contain both classes (promoter and non-promoter examples)." },
      { status: 400 }
    );
  }

  const maxLen = Math.max(...rows.map((r) => r.sequence.length));
  if (maxLen > 2000) {
    return NextResponse.json(
      { error: "Sequences are too long for this demo (max 2000 bases). Trim your dataset." },
      { status: 400 }
    );
  }

  // --- Encode ---
  const encoded = rows.map((r) => oneHotEncode(r.sequence, maxLen));
  const flat = flattenBatch(encoded);
  const labels = new Float32Array(rows.map((r) => r.label));

  // --- Shuffle + split ---
  const order = shuffleIndices(rows.length);
  const valCount = Math.max(1, Math.round(rows.length * (hyperparams.valSplit ?? 0.2)));
  const valIdx = new Set(order.slice(0, valCount));

  const trainX: number[] = [];
  const trainY: number[] = [];
  const valX: number[] = [];
  const valY: number[] = [];

  for (let i = 0; i < rows.length; i++) {
    const start = i * maxLen * 4;
    const slice = Array.from(flat.slice(start, start + maxLen * 4));
    if (valIdx.has(i)) {
      valX.push(...slice);
      valY.push(labels[i]);
    } else {
      trainX.push(...slice);
      trainY.push(labels[i]);
    }
  }

  const nTrain = trainY.length;
  const nVal = valY.length;

  if (nTrain < 4) {
    return NextResponse.json(
      { error: "Not enough training examples after the validation split. Add more data or lower the validation split." },
      { status: 400 }
    );
  }

  const xsTrain = tf.tensor3d(trainX, [nTrain, maxLen, 4]);
  const ysTrain = tf.tensor2d(trainY, [nTrain, 1]);
  const xsVal = nVal > 0 ? tf.tensor3d(valX, [nVal, maxLen, 4]) : null;
  const ysVal = nVal > 0 ? tf.tensor2d(valY, [nVal, 1]) : null;

  const model = buildModel(maxLen, hyperparams);

  const history: Array<{
    epoch: number;
    loss: number;
    acc: number;
    val_loss: number | null;
    val_acc: number | null;
  }> = [];

  try {
    await model.fit(xsTrain, ysTrain, {
      epochs: hyperparams.epochs,
      batchSize: hyperparams.batchSize,
      validationData: xsVal && ysVal ? [xsVal, ysVal] : undefined,
      verbose: 0,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          history.push({
            epoch: epoch + 1,
            loss: logs?.loss ?? 0,
            acc: (logs?.acc as number) ?? (logs?.accuracy as number) ?? 0,
            val_loss: logs?.val_loss ?? null,
            val_acc: (logs?.val_acc as number) ?? (logs?.val_accuracy as number) ?? null,
          });
        },
      },
    });
  } catch (err) {
    tf.dispose([xsTrain, ysTrain, xsVal, ysVal].filter(Boolean) as tf.Tensor[]);
    return NextResponse.json(
      { error: `Training failed: ${(err as Error).message}` },
      { status: 500 }
    );
  }

  // --- Evaluate on validation split for a confusion matrix ---
  let confusionMatrix = { tp: 0, tn: 0, fp: 0, fn: 0 };
  let metrics = { accuracy: 0, precision: 0, recall: 0, f1: 0 };

  if (xsVal && ysVal && nVal > 0) {
    const preds = model.predict(xsVal) as tf.Tensor;
    const predArr = Array.from(await preds.data());
    const trueArr = valY;

    let tp = 0,
      tn = 0,
      fp = 0,
      fn = 0;
    for (let i = 0; i < predArr.length; i++) {
      const pred = predArr[i] >= 0.5 ? 1 : 0;
      const actual = trueArr[i];
      if (pred === 1 && actual === 1) tp++;
      else if (pred === 0 && actual === 0) tn++;
      else if (pred === 1 && actual === 0) fp++;
      else fn++;
    }
    confusionMatrix = { tp, tn, fp, fn };
    const accuracy = (tp + tn) / (tp + tn + fp + fn || 1);
    const precision = tp / (tp + fp || 1);
    const recall = tp / (tp + fn || 1);
    const f1 = (2 * precision * recall) / (precision + recall || 1);
    metrics = { accuracy, precision, recall, f1 };
    preds.dispose();
  }

  // --- Serialize the trained model to send back to the browser ---
  let modelTopology: unknown = null;
  let weightSpecs: unknown = null;
  let weightDataB64 = "";

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      modelTopology = artifacts.modelTopology;
      weightSpecs = artifacts.weightSpecs;
      const buf = Buffer.from(artifacts.weightData as ArrayBuffer);
      weightDataB64 = buf.toString("base64");
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    })
  );

  tf.dispose([xsTrain, ysTrain, xsVal, ysVal].filter(Boolean) as tf.Tensor[]);
  model.dispose();

  return NextResponse.json({
    history,
    metrics,
    confusionMatrix,
    dataset: { total: rows.length, nTrain, nVal, positives, negatives, maxLen },
    model: { modelTopology, weightSpecs, weightDataB64 },
  });
}
