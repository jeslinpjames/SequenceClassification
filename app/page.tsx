"use client";

import { useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { validateSequence, DatasetRow } from "@/lib/dna";
import { DEFAULT_HYPERPARAMS, HyperParams } from "@/lib/modelBuilder";
import SequenceViewer from "@/components/SequenceViewer";
import OneHotViewer from "@/components/OneHotViewer";
import DatasetPanel from "@/components/DatasetPanel";
import ModelConfig from "@/components/ModelConfig";
import TrainingDashboard, { EpochLog, Metrics, ConfusionMatrix } from "@/components/TrainingDashboard";
import InferencePanel from "@/components/InferencePanel";

const SAMPLE_SEQUENCE = "TTGACAATTAATCATCGAACTAGTTAACTAGTACGCAAGTTCACGTAAAAAGGGTATCG";

export default function Home() {
  // --- Sequence display / one-hot ---
  const [rawInput, setRawInput] = useState(SAMPLE_SEQUENCE);
  const [showOneHot, setShowOneHot] = useState(false);
  const validation = validateSequence(rawInput);

  // --- Dataset ---
  const [dataset, setDataset] = useState<DatasetRow[]>([]);

  // --- Hyperparameters ---
  const [hp, setHp] = useState<HyperParams>(DEFAULT_HYPERPARAMS);

  // --- Training state ---
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrix | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [modelMaxLen, setModelMaxLen] = useState(0);
  const playbackTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  async function handleTrain() {
    setTrainError(null);
    setMetrics(null);
    setConfusionMatrix(null);
    setHistory([]);
    setModel(null);

    if (dataset.length < 10) {
      setTrainError("Upload a dataset with at least 10 labeled sequences first.");
      return;
    }

    setIsTraining(true);
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: dataset, hyperparams: hp }),
      });
      const data = await res.json();
      if (!res.ok) {
        setTrainError(data.error || "Training failed.");
        setIsTraining(false);
        return;
      }

      // Play back the (already-computed) epoch history to give a live feel,
      // since the actual training already ran to completion on the server.
      const fullHistory: EpochLog[] = data.history;
      let i = 0;
      playbackTimer.current = setInterval(() => {
        i++;
        setHistory(fullHistory.slice(0, i));
        if (i >= fullHistory.length) {
          if (playbackTimer.current) clearInterval(playbackTimer.current);
          setMetrics(data.metrics);
          setConfusionMatrix(data.confusionMatrix);
          setIsTraining(false);
          loadModel(data.model, data.dataset.maxLen);
        }
      }, Math.max(15, Math.min(120, 3000 / fullHistory.length)));
    } catch (err) {
      setTrainError((err as Error).message);
      setIsTraining(false);
    }
  }

  async function loadModel(
    modelArtifacts: { modelTopology: unknown; weightSpecs: unknown; weightDataB64: string },
    maxLen: number
  ) {
    const binary = atob(modelArtifacts.weightDataB64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    const loaded = await tf.loadLayersModel(
      tf.io.fromMemory({
        modelTopology: modelArtifacts.modelTopology as tf.io.ModelJSON["modelTopology"],
        weightSpecs: modelArtifacts.weightSpecs as tf.io.WeightsManifestEntry[],
        weightData: bytes.buffer,
      })
    );
    setModel(loaded);
    setModelMaxLen(maxLen);
  }

  return (
    <main className="min-h-screen bg-lab-bg text-lab-ink">
      <header className="border-b border-lab-border px-6 py-4">
        <h1 className="text-lg font-semibold">DNA Sequence Classifier</h1>
        <p className="text-sm text-lab-dim">
          Encode a sequence, train a recurrent model on your data, and test it — all live, no
          setup required.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-4 max-w-[1400px] mx-auto">
        {/* Column 1: Sequence + Dataset */}
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">
            1 · Sequence &amp; data
          </h2>

          <div className="space-y-2">
            <textarea
              value={rawInput}
              onChange={(e) => setRawInput(e.target.value)}
              placeholder="Paste a DNA sequence (A, C, G, T)…"
              className="w-full h-20 bg-lab-bg border border-lab-border rounded-md p-2 text-sm font-mono resize-none focus:outline-none focus:border-accent"
            />
            {validation.errors.length > 0 && (
              <p className="text-xs text-base-T">{validation.errors[0]}</p>
            )}
            {validation.valid && (
              <>
                <SequenceViewer sequence={validation.cleaned} />
                <button
                  onClick={() => setShowOneHot((v) => !v)}
                  className="text-xs px-3 py-1.5 rounded-md bg-lab-panel2 border border-lab-border hover:border-accent transition-colors"
                >
                  {showOneHot ? "Hide" : "Show"} one-hot encoding
                </button>
                {showOneHot && <OneHotViewer sequence={validation.cleaned} />}
              </>
            )}
          </div>

          <div className="border-t border-lab-border pt-4 space-y-2">
            <h3 className="text-xs font-semibold text-lab-dim uppercase tracking-wide">
              Training dataset
            </h3>
            <DatasetPanel onDataset={setDataset} />
          </div>
        </section>

        {/* Column 2: Model + Training */}
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">
            2 · Model &amp; training
          </h2>

          <ModelConfig hp={hp} onChange={setHp} disabled={isTraining} />

          <button
            onClick={handleTrain}
            disabled={isTraining || dataset.length < 10}
            className="w-full py-2 rounded-md bg-accent text-lab-bg font-medium text-sm disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90 transition-colors"
          >
            {isTraining ? "Training…" : "Train model"}
          </button>

          {trainError && <p className="text-xs text-base-T">{trainError}</p>}

          <TrainingDashboard
            history={history}
            totalEpochs={hp.epochs}
            metrics={metrics}
            confusionMatrix={confusionMatrix}
            isTraining={isTraining}
          />
        </section>

        {/* Column 3: Inference */}
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">
            3 · Try it
          </h2>
          <InferencePanel model={model} maxLen={modelMaxLen} />
          {!model && (
            <p className="text-xs text-lab-dim">
              Once training finishes, the trained model loads here automatically — predictions
              run instantly in your browser, no server round-trip.
            </p>
          )}
        </section>
      </div>
    </main>
  );
}
