import { NextRequest, NextResponse } from "next/server";
import * as tf from "@tensorflow/tfjs";
import { oneHotEncode, flattenBatch, resolveAlphabet, validateSequence, SequenceTypeConfig } from "@/lib/sequence";
import { buildModel, HyperParams } from "@/lib/modelBuilder";

export const runtime = "nodejs";
export const maxDuration = 60; // seconds — raise on Vercel Pro if you need longer runs

interface ClassInput {
  name: string;
  sequences: string[];
}

interface TrainRequestBody {
  classes: ClassInput[];
  sequenceType: SequenceTypeConfig;
  hyperparams: HyperParams;
}

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

  const { classes, sequenceType, hyperparams } = body;

  if (!Array.isArray(classes) || classes.length < 2) {
    return NextResponse.json({ error: "Add at least 2 classes to train on." }, { status: 400 });
  }

  const classNames = classes.map((c, i) => c.name?.trim() || `Class ${i + 1}`);
  const allSequences = classes.flatMap((c) => c.sequences);

  if (allSequences.length < 10) {
    return NextResponse.json(
      { error: "Add at least 10 example sequences in total across your classes." },
      { status: 400 }
    );
  }
  for (const c of classes) {
    if (c.sequences.length < 3) {
      return NextResponse.json(
        { error: `Each class needs at least 3 examples ("${c.name}" has ${c.sequences.length}).` },
        { status: 400 }
      );
    }
  }

  // --- Resolve the alphabet (DNA / RNA / protein / auto-detected / custom) ---
  const alphabet = resolveAlphabet(sequenceType, allSequences);
  if (alphabet.length < 2) {
    return NextResponse.json(
      { error: "Couldn't determine an alphabet with at least 2 symbols from your data." },
      { status: 400 }
    );
  }

  // --- Validate + collect rows ---
  const rows: { sequence: string; label: number }[] = [];
  const errors: string[] = [];
  classes.forEach((c, classIdx) => {
    for (const raw of c.sequences) {
      const { valid, cleaned, errors: seqErrors } = validateSequence(raw, alphabet);
      if (!valid) {
        errors.push(`"${c.name}": ${seqErrors[0]}`);
        continue;
      }
      rows.push({ sequence: cleaned, label: classIdx });
    }
  });

  if (rows.length < 10) {
    return NextResponse.json(
      { error: `Too many invalid sequences to train. First issue: ${errors[0] ?? "unknown"}` },
      { status: 400 }
    );
  }

  const maxLen = Math.max(...rows.map((r) => r.sequence.length));
  if (maxLen > 2000) {
    return NextResponse.json(
      { error: "Sequences are too long for this demo (max 2000 symbols). Trim your data." },
      { status: 400 }
    );
  }

  const vocabSize = alphabet.length;
  const numClasses = classes.length;

  // --- Encode ---
  const encoded = rows.map((r) => oneHotEncode(r.sequence, alphabet, maxLen));
  const flat = flattenBatch(encoded, vocabSize);
  const labels = rows.map((r) => r.label);

  // --- Shuffle + split ---
  const order = shuffleIndices(rows.length);
  const valCount = Math.max(numClasses, Math.round(rows.length * (hyperparams.valSplit ?? 0.2)));
  const valIdx = new Set(order.slice(0, valCount));

  const trainX: number[] = [];
  const trainY: number[] = [];
  const valX: number[] = [];
  const valY: number[] = [];

  const stride = maxLen * vocabSize;
  for (let i = 0; i < rows.length; i++) {
    const start = i * stride;
    const slice = Array.from(flat.slice(start, start + stride));
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

  if (nTrain < numClasses * 2) {
    return NextResponse.json(
      { error: "Not enough training examples after the validation split. Add more data or lower the validation split." },
      { status: 400 }
    );
  }

  const xsTrain = tf.tensor3d(trainX, [nTrain, maxLen, vocabSize]);
  const ysTrain = tf.tensor2d(trainY, [nTrain, 1]);
  const xsVal = nVal > 0 ? tf.tensor3d(valX, [nVal, maxLen, vocabSize]) : null;
  const ysVal = nVal > 0 ? tf.tensor2d(valY, [nVal, 1]) : null;

  let model: tf.LayersModel;
  try {
    model = buildModel(maxLen, vocabSize, numClasses, hyperparams);
  } catch (err) {
    tf.dispose([xsTrain, ysTrain, xsVal, ysVal].filter(Boolean) as tf.Tensor[]);
    return NextResponse.json({ error: `Couldn't build model: ${(err as Error).message}` }, { status: 500 });
  }

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
    model.dispose();
    return NextResponse.json({ error: `Training failed: ${(err as Error).message}` }, { status: 500 });
  }

  // --- Evaluate on validation split: overall accuracy, macro precision/recall/f1, NxN confusion matrix ---
  let confusionMatrix: number[][] = [];
  let metrics = { accuracy: 0, precision: 0, recall: 0, f1: 0 };

  if (xsVal && ysVal && nVal > 0) {
    const preds = model.predict(xsVal) as tf.Tensor;
    const predClasses = Array.from(await preds.argMax(-1).data());
    const trueClasses = valY;

    confusionMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
    for (let i = 0; i < predClasses.length; i++) {
      confusionMatrix[trueClasses[i]][predClasses[i]]++;
    }

    let correct = 0;
    const perClass = { precision: 0, recall: 0, f1: 0 };
    for (let c = 0; c < numClasses; c++) {
      const tp = confusionMatrix[c][c];
      const fp = confusionMatrix.reduce((s, row, r) => (r === c ? s : s + row[c]), 0);
      const fn = confusionMatrix[c].reduce((s, v, cc) => (cc === c ? s : s + v), 0);
      const precision = tp / (tp + fp || 1);
      const recall = tp / (tp + fn || 1);
      const f1 = (2 * precision * recall) / (precision + recall || 1);
      perClass.precision += precision;
      perClass.recall += recall;
      perClass.f1 += f1;
      correct += tp;
    }
    metrics = {
      accuracy: correct / (predClasses.length || 1),
      precision: perClass.precision / numClasses,
      recall: perClass.recall / numClasses,
      f1: perClass.f1 / numClasses,
    };
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
        modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" },
      };
    })
  );

  tf.dispose([xsTrain, ysTrain, xsVal, ysVal].filter(Boolean) as tf.Tensor[]);
  model.dispose();

  return NextResponse.json({
    history,
    metrics,
    confusionMatrix,
    classNames,
    alphabet,
    dataset: { total: rows.length, nTrain, nVal, maxLen, vocabSize, numClasses },
    warnings: errors.slice(0, 5),
    model: { modelTopology, weightSpecs, weightDataB64 },
  });
}
