import * as tf from "@tensorflow/tfjs";

export type ModelType = "rnn" | "lstm" | "gru" | "bilstm";

export interface HyperParams {
  modelType: ModelType;
  hiddenUnits: number;
  numLayers: number;
  dropout: number;
  learningRate: number;
  batchSize: number;
  epochs: number;
  valSplit: number;
  seed: number;
  crossValidate: boolean;
  folds: number;
}

export const DEFAULT_HYPERPARAMS: HyperParams = {
  modelType: "gru",
  hiddenUnits: 16,
  numLayers: 1,
  dropout: 0.2,
  learningRate: 0.01,
  batchSize: 8,
  epochs: 15,
  valSplit: 0.2,
  seed: 42,
  crossValidate: false,
  folds: 5,
};

function makeRecurrentLayer(
  type: ModelType,
  units: number,
  returnSequences: boolean,
  seed: number
): tf.layers.Layer {
  const kernelInitializer = tf.initializers.glorotUniform({ seed });
  const recurrentInitializer = tf.initializers.orthogonal({ seed: seed + 1 });
  const common = { units, returnSequences, dropout: 0, kernelInitializer, recurrentInitializer } as const;

  switch (type) {
    case "rnn":
      return tf.layers.simpleRNN(common);
    case "lstm":
      return tf.layers.lstm(common);
    case "gru":
      return tf.layers.gru(common);
    case "bilstm":
      return tf.layers.bidirectional({
        layer: tf.layers.lstm({ units, returnSequences, kernelInitializer, recurrentInitializer }) as tf.layers.RNN,
        mergeMode: "concat",
      });
    default:
      throw new Error(`Unknown model type: ${type}`);
  }
}

/**
 * Builds a small sequential recurrent classifier:
 * Input [seqLen, vocabSize] -> (recurrent layer(s) + dropout) -> Dense(numClasses, softmax)
 *
 * Every weight-bearing layer gets a seeded initializer, derived from hp.seed, so the same
 * seed always produces the same starting weights — needed to fairly compare architectures
 * instead of confounding "which model is better" with "which model got luckier init."
 */
export function buildModel(
  seqLen: number,
  vocabSize: number,
  numClasses: number,
  hp: HyperParams
): tf.LayersModel {
  const model = tf.sequential();

  for (let layerIdx = 0; layerIdx < hp.numLayers; layerIdx++) {
    const isLast = layerIdx === hp.numLayers - 1;
    const layerSeed = hp.seed + layerIdx * 10;
    const layer = makeRecurrentLayer(hp.modelType, hp.hiddenUnits, !isLast, layerSeed);
    if (layerIdx === 0) {
      model.add(tf.layers.inputLayer({ inputShape: [seqLen, vocabSize] }));
    }
    model.add(layer);
    if (hp.dropout > 0) {
      model.add(tf.layers.dropout({ rate: hp.dropout, seed: hp.seed + 999 }));
    }
  }

  model.add(
    tf.layers.dense({
      units: 8,
      activation: "relu",
      kernelInitializer: tf.initializers.glorotUniform({ seed: hp.seed + 500 }),
    })
  );
  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: "softmax",
      kernelInitializer: tf.initializers.glorotUniform({ seed: hp.seed + 501 }),
    })
  );

  model.compile({
    optimizer: tf.train.adam(hp.learningRate),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
