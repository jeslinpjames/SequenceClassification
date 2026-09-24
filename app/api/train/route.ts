import { NextRequest, NextResponse } from "next/server";
import * as tf from "@tensorflow/tfjs";
import { oneHotEncode, flattenBatch, resolveAlphabet, validateSequence, SequenceTypeConfig } from "@/lib/sequence";
import { buildModel, HyperParams } from "@/lib/modelBuilder";
import { mulberry32, seededShuffle, stratifiedFolds } from "@/lib/seededRandom";

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

interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  val_loss: number | null;
  val_acc: number | null;
}

function computeConfusionAndMetrics(
  predClasses: number[],
  trueClasses: number[],
  numClasses: number
) {
  const confusionMatrix: number[][] = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
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
  return {
    confusionMatrix,
    metrics: {
      accuracy: correct / (predClasses.length || 1),
      precision: perClass.precision / numClasses,
      recall: perClass.recall / numClasses,
      f1: perClass.f1 / numClasses,
    },
  };
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}
function std(xs: number[]): number {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

export async function POST(req: NextRequest) {
  let body: TrainRequestBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const { classes, sequenceType, hyperparams: hp } = body;

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
    if (hp.crossValidate && c.sequences.length < hp.folds) {
      return NextResponse.json(
        {
          error: `With ${hp.folds}-fold cross-validation, every class needs at least ${hp.folds} examples ("${c.name}" has ${c.sequences.length}). Add more examples, or lower the fold count.`,
        },
        { status: 400 }
      );
    }
  }

  const alphabet = resolveAlphabet(sequenceType, allSequences);
  if (alphabet.length < 2) {
    return NextResponse.json(
      { error: "Couldn't determine an alphabet with at least 2 symbols from your data." },
      { status: 400 }
    );
  }

  const rows: { sequence: string; label: number }[] = [];
  const parseErrors: string[] = [];
  classes.forEach((c, classIdx) => {
    for (const raw of c.sequences) {
      const { valid, cleaned, errors: seqErrors } = validateSequence(raw, alphabet);
      if (!valid) {
        parseErrors.push(`"${c.name}": ${seqErrors[0]}`);
        continue;
      }
      rows.push({ sequence: cleaned, label: classIdx });
    }
  });

  if (rows.length < 10) {
    return NextResponse.json(
      { error: `Too many invalid sequences to train. First issue: ${parseErrors[0] ?? "unknown"}` },
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
  const labels = rows.map((r) => r.label);
  const encoded = rows.map((r) => oneHotEncode(r.sequence, alphabet, maxLen));
  const flat = flattenBatch(encoded, vocabSize);
  const stride = maxLen * vocabSize;

  function tensorsFor(indices: number[]) {
    const x: number[] = [];
    const y: number[] = [];
    for (const i of indices) {
      const start = i * stride;
      x.push(...Array.from(flat.slice(start, start + stride)));
      y.push(labels[i]);
    }
    return {
      xs: tf.tensor3d(x, [indices.length, maxLen, vocabSize]),
      ys: tf.tensor2d(y, [indices.length, 1]),
      rawY: y,
    };
  }

  async function trainOneModel(trainIdx: number[], valIdx: number[] | null) {
    const shuffledTrainIdx = seededShuffle(trainIdx, rng);
    const { xs: xsTrain, ys: ysTrain } = tensorsFor(shuffledTrainIdx);
    const val = valIdx && valIdx.length > 0 ? tensorsFor(valIdx) : null;

    const model = buildModel(maxLen, vocabSize, numClasses, hp);
    const history: EpochLog[] = [];

    await model.fit(xsTrain, ysTrain, {
      epochs: hp.epochs,
      batchSize: hp.batchSize,
      validationData: val ? [val.xs, val.ys] : undefined,
      shuffle: false, // we pre-shuffle everything ourselves with a seeded RNG; tfjs's
      // own internal shuffle uses unseeded randomness and would silently break reproducibility
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

    let evalResult: ReturnType<typeof computeConfusionAndMetrics> | null = null;
    if (val) {
      const preds = model.predict(val.xs) as tf.Tensor;
      const predClasses = Array.from(await preds.argMax(-1).data());
      evalResult = computeConfusionAndMetrics(predClasses, val.rawY, numClasses);
      preds.dispose();
      val.xs.dispose();
      val.ys.dispose();
    }

    xsTrain.dispose();
    ysTrain.dispose();

    return { model, history, evalResult };
  }

  const rng = mulberry32(hp.seed);
  let history: EpochLog[] = [];
  let metrics = { accuracy: 0, precision: 0, recall: 0, f1: 0 };
  let confusionMatrix: number[][] = [];
  let cvFoldAccuracies: number[] | null = null;
  let cvStd: number | null = null;
  let finalModel: tf.LayersModel;

  try {
    if (hp.crossValidate) {
      const folds = stratifiedFolds(labels, hp.folds, rng);
      const foldMetrics: { accuracy: number; precision: number; recall: number; f1: number }[] = [];
      let summedConfusion: number[][] | null = null;
      const epochSums: { loss: number; acc: number }[] = [];

      for (let f = 0; f < folds.length; f++) {
        const valIdx = folds[f];
        const trainIdx = folds.filter((_, i) => i !== f).flat();
        const { model, history: foldHistory, evalResult } = await trainOneModel(trainIdx, valIdx);

        if (evalResult) {
          foldMetrics.push(evalResult.metrics);
          if (!summedConfusion) {
            summedConfusion = evalResult.confusionMatrix.map((row) => [...row]);
          } else {
            evalResult.confusionMatrix.forEach((row, r) => row.forEach((v, c) => (summedConfusion![r][c] += v)));
          }
        }
        foldHistory.forEach((h, i) => {
          if (!epochSums[i]) epochSums[i] = { loss: 0, acc: 0 };
          epochSums[i].loss += h.loss;
          epochSums[i].acc += h.acc;
        });

        model.dispose();
      }

      history = epochSums.map((sum, i) => ({
        epoch: i + 1,
        loss: sum.loss / folds.length,
        acc: sum.acc / folds.length,
        val_loss: null,
        val_acc: null,
      }));
      cvFoldAccuracies = foldMetrics.map((m) => m.accuracy);
      cvStd = std(cvFoldAccuracies);
      metrics = {
        accuracy: mean(foldMetrics.map((m) => m.accuracy)),
        precision: mean(foldMetrics.map((m) => m.precision)),
        recall: mean(foldMetrics.map((m) => m.recall)),
        f1: mean(foldMetrics.map((m) => m.f1)),
      };
      confusionMatrix = summedConfusion ?? [];

      // Train the model that actually gets shipped to the browser on ALL data,
      // now that cross-validation has given us a reliable performance estimate.
      const allIdx = Array.from({ length: rows.length }, (_, i) => i);
      const { model } = await trainOneModel(allIdx, null);
      finalModel = model;
    } else {
      // Single stratified, seeded split (still far more stable than a plain random one).
      const perClassIdx = new Map<number, number[]>();
      labels.forEach((label, idx) => {
        if (!perClassIdx.has(label)) perClassIdx.set(label, []);
        perClassIdx.get(label)!.push(idx);
      });
      const trainIdx: number[] = [];
      const valIdx: number[] = [];
      for (const indices of perClassIdx.values()) {
        const shuffled = seededShuffle(indices, rng);
        const nVal = Math.max(1, Math.round(shuffled.length * (hp.valSplit ?? 0.2)));
        valIdx.push(...shuffled.slice(0, nVal));
        trainIdx.push(...shuffled.slice(nVal));
      }

      const { model, history: h, evalResult } = await trainOneModel(trainIdx, valIdx);
      history = h;
      if (evalResult) {
        metrics = evalResult.metrics;
        confusionMatrix = evalResult.confusionMatrix;
      }
      finalModel = model;
    }
  } catch (err) {
    return NextResponse.json({ error: `Training failed: ${(err as Error).message}` }, { status: 500 });
  }

  let modelTopology: unknown = null;
  let weightSpecs: unknown = null;
  let weightDataB64 = "";

  await finalModel.save(
    tf.io.withSaveHandler(async (artifacts) => {
      modelTopology = artifacts.modelTopology;
      weightSpecs = artifacts.weightSpecs;
      const buf = Buffer.from(artifacts.weightData as ArrayBuffer);
      weightDataB64 = buf.toString("base64");
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    })
  );
  finalModel.dispose();

  return NextResponse.json({
    history,
    metrics,
    confusionMatrix,
    classNames,
    alphabet,
    crossValidated: hp.crossValidate,
    cvFoldAccuracies,
    cvStd,
    dataset: { total: rows.length, maxLen, vocabSize, numClasses },
    warnings: parseErrors.slice(0, 5),
    model: { modelTopology, weightSpecs, weightDataB64 },
  });
}
