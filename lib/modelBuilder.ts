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
}

export const DEFAULT_HYPERPARAMS: HyperParams = {
  modelType: "gru",
  hiddenUnits: 16,
  numLayers: 1,
  dropout: 0.2,
  learningRate: 0.01,
  batchSize: 8,
  epochs: 30,
  valSplit: 0.2,
};

function makeRecurrentLayer(
  type: ModelType,
  units: number,
  returnSequences: boolean
): tf.layers.Layer {
  const common = { units, returnSequences, dropout: 0 } as const;
  switch (type) {
    case "rnn":
      return tf.layers.simpleRNN(common);
    case "lstm":
      return tf.layers.lstm(common);
    case "gru":
      return tf.layers.gru(common);
    case "bilstm":
      return tf.layers.bidirectional({
        layer: tf.layers.lstm({ units, returnSequences }) as tf.layers.RNN,
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
 * Uses softmax + sparseCategoricalCrossentropy uniformly for 2-or-more classes, so binary
 * and multi-class (Teachable-Machine-style "any number of classes") share one code path.
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
    const layer = makeRecurrentLayer(hp.modelType, hp.hiddenUnits, !isLast);
    if (layerIdx === 0) {
      model.add(tf.layers.inputLayer({ inputShape: [seqLen, vocabSize] }));
    }
    model.add(layer);
    if (hp.dropout > 0) {
      model.add(tf.layers.dropout({ rate: hp.dropout }));
    }
  }

  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(hp.learningRate),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
