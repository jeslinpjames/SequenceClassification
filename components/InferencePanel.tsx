"use client";

import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { oneHotEncode, validateSequence, colorForSymbol } from "@/lib/sequence";

export default function InferencePanel({
  model,
  maxLen,
  alphabet,
  classNames,
}: {
  model: tf.LayersModel | null;
  maxLen: number;
  alphabet: string[];
  classNames: string[];
}) {
  const [input, setInput] = useState("");
  const [probs, setProbs] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  function runInference() {
    setError(null);
    setProbs(null);
    if (!model) {
      setError("Train a model first.");
      return;
    }
    const { valid, cleaned, errors } = validateSequence(input, alphabet);
    if (!valid) {
      setError(errors[0]);
      return;
    }

    const matrix = oneHotEncode(cleaned, alphabet, maxLen);
    const flat = matrix.flat();
    const x = tf.tensor3d(flat, [1, maxLen, alphabet.length]);
    const pred = model.predict(x) as tf.Tensor;
    pred.data().then((data) => {
      setProbs(Array.from(data));
      x.dispose();
      pred.dispose();
    });
  }

  // Deterministic-ish colors for class bars, distinct from base symbol colors.
  const classColor = (i: number) =>
    ["#2DD4BF", "#FB7185", "#FBBF24", "#38BDF8", "#A78BFA", "#4ADE80", "#F472B6", "#FB923C"][
      i % 8
    ];

  return (
    <div className="space-y-3">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={
          model
            ? `Paste a sequence (will be padded/truncated to ${maxLen} symbols)…`
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

      {probs && (
        <div className="space-y-2">
          {classNames.map((name, i) => {
            const pct = Math.round(probs[i] * 100);
            return (
              <div key={name} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span style={{ color: classColor(i) }}>{name}</span>
                  <span className="font-mono">{pct}%</span>
                </div>
                <div className="h-2 bg-lab-panel2 rounded-full overflow-hidden">
                  <div
                    className="h-full transition-all duration-300"
                    style={{ width: `${pct}%`, backgroundColor: classColor(i) }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
