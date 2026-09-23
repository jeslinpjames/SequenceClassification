"use client";

import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { oneHotEncode, validateSequence } from "@/lib/dna";

export default function InferencePanel({
  model,
  maxLen,
}: {
  model: tf.LayersModel | null;
  maxLen: number;
}) {
  const [input, setInput] = useState("");
  const [prob, setProb] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  function runInference() {
    setError(null);
    setProb(null);
    if (!model) {
      setError("Train a model first.");
      return;
    }
    const { valid, cleaned, errors } = validateSequence(input);
    if (!valid) {
      setError(errors[0]);
      return;
    }

    const matrix = oneHotEncode(cleaned, maxLen);
    const flat = matrix.flat();
    const x = tf.tensor3d(flat, [1, maxLen, 4]);
    const pred = model.predict(x) as tf.Tensor;
    pred.data().then((data) => {
      setProb(data[0]);
      x.dispose();
      pred.dispose();
    });
  }

  const promoterPct = prob !== null ? Math.round(prob * 100) : null;

  return (
    <div className="space-y-3">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={
          model
            ? `Paste a sequence (will be padded/truncated to ${maxLen} bases)…`
            : "Train a model on the left first…"
        }
        className="w-full h-20 bg-lab-bg border border-lab-border rounded-md p-2 text-sm font-mono resize-none focus:outline-none focus:border-accent"
        disabled={!model}
      />
      <button
        onClick={runInference}
        disabled={!model || input.trim().length === 0}
        className="w-full py-2 rounded-md bg-accent text-lab-bg font-medium text-sm disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90 transition-colors"
      >
        Predict
      </button>

      {error && <p className="text-xs text-base-T">{error}</p>}

      {promoterPct !== null && (
        <div className="space-y-2">
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="base-a">Promoter</span>
              <span className="font-mono">{promoterPct}%</span>
            </div>
            <div className="h-2 bg-lab-panel2 rounded-full overflow-hidden">
              <div className="h-full bg-base-A transition-all duration-300" style={{ width: `${promoterPct}%` }} />
            </div>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="base-t">Non-promoter</span>
              <span className="font-mono">{100 - promoterPct}%</span>
            </div>
            <div className="h-2 bg-lab-panel2 rounded-full overflow-hidden">
              <div
                className="h-full bg-base-T transition-all duration-300"
                style={{ width: `${100 - promoterPct}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
