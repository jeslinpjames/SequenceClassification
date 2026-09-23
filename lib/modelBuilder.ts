import * as tf from "@tensorflow/tfjs";

export type ModelType = "rnn" | "lstm" | "gru" | "bilstm";

export interface HyperParams {
  modelType: ModelType;
  hiddenUnits: number; // units per recurrent layer
  numLayers: number; // 1-3
  dropout: number; // 0-0.6
  learningRate: number;
  batchSize: number;
  epochs: number;
  valSplit: number; // 0.1 - 0.4
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
 * Input [seqLen, 4] -> (recurrent layer(s) + dropout) -> Dense(1, sigmoid)
 */
export function buildModel(seqLen: number, hp: HyperParams): tf.LayersModel {
  const model = tf.sequential();

  for (let layerIdx = 0; layerIdx < hp.numLayers; layerIdx++) {
    const isLast = layerIdx === hp.numLayers - 1;
    const layer = makeRecurrentLayer(hp.modelType, hp.hiddenUnits, !isLast);
    if (layerIdx === 0) {
      model.add(tf.layers.inputLayer({ inputShape: [seqLen, 4] }));
    }
    model.add(layer);
    if (hp.dropout > 0) {
      model.add(tf.layers.dropout({ rate: hp.dropout }));
    }
  }

  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(hp.learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

export function countParams(model: tf.LayersModel): number {
  return model.countParams();
}
