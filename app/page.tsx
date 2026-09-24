"use client";

import { useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  validateSequence,
  resolveAlphabet,
  SequenceTypeConfig,
} from "@/lib/sequence";
import { DEFAULT_HYPERPARAMS, HyperParams } from "@/lib/modelBuilder";
import SequenceViewer from "@/components/SequenceViewer";
import OneHotViewer from "@/components/OneHotViewer";
import SequenceTypeSelector from "@/components/SequenceTypeSelector";
import ClassesPanel, { SeqClass, makeInitialClasses } from "@/components/ClassesPanel";
import ModelConfig from "@/components/ModelConfig";
import TrainingDashboard, { EpochLog, Metrics } from "@/components/TrainingDashboard";
import InferencePanel from "@/components/InferencePanel";

const SAMPLE_SEQUENCE = "TTGACAATTAATCATCGAACTAGTTAACTAGTACGCAAGTTCACGTAAAAAGGGTATCG";

export default function Home() {
  // --- Sequence type / alphabet (shared by the encoding preview and training) ---
  const [seqType, setSeqType] = useState<SequenceTypeConfig>({ preset: "dna" });

  // --- Encoding preview (kept from the original single-sequence view) ---
  const [rawInput, setRawInput] = useState(SAMPLE_SEQUENCE);
  const [showOneHot, setShowOneHot] = useState(false);
  const previewAlphabet = resolveAlphabet(seqType, [rawInput]);
  const validation = validateSequence(rawInput, previewAlphabet);

  // --- Classes (Teachable-Machine-style data) ---
  const [classes, setClasses] = useState<SeqClass[]>(makeInitialClasses());
  const totalExamples = classes.reduce((s, c) => s + c.sequences.length, 0);

  // --- Hyperparameters ---
  const [hp, setHp] = useState<HyperParams>(DEFAULT_HYPERPARAMS);

  // --- Training state ---
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([]);
  const [trainClassNames, setTrainClassNames] = useState<string[]>([]);
  const [crossValidated, setCrossValidated] = useState(false);
  const [cvFoldAccuracies, setCvFoldAccuracies] = useState<number[] | null>(null);
  const [cvStd, setCvStd] = useState<number | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [modelMaxLen, setModelMaxLen] = useState(0);
  const [modelAlphabet, setModelAlphabet] = useState<string[]>([]);
  const playbackTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  async function handleTrain() {
    setTrainError(null);
    setWarnings([]);
    setMetrics(null);
    setConfusionMatrix([]);
    setHistory([]);
    setModel(null);
    setCvFoldAccuracies(null);
    setCvStd(null);

    if (totalExamples < 10) {
      setTrainError("Add at least 10 example sequences in total across your classes.");
      return;
    }
    if (classes.some((c) => c.sequences.length < 3)) {
      setTrainError("Each class needs at least 3 examples.");
      return;
    }

    setIsTraining(true);
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          classes: classes.map((c) => ({ name: c.name, sequences: c.sequences })),
          sequenceType: seqType,
          hyperparams: hp,
        }),
      });

      let data: any;
      try {
        data = await res.json();
      } catch {
        setTrainError(
          "The server didn't respond in time — this usually means training took too long for the current hosting limits. Try fewer epochs, fewer/smaller layers, or a smaller dataset."
        );
        setIsTraining(false);
        return;
      }

      if (!res.ok) {
        setTrainError(data.error || "Training failed.");
        setIsTraining(false);
        return;
      }

      if (data.warnings?.length) setWarnings(data.warnings);

      const fullHistory: EpochLog[] = data.history;
      let i = 0;
      playbackTimer.current = setInterval(() => {
        i++;
        setHistory(fullHistory.slice(0, i));
        if (i >= fullHistory.length) {
          if (playbackTimer.current) clearInterval(playbackTimer.current);
          setMetrics(data.metrics);
          setConfusionMatrix(data.confusionMatrix);
          setTrainClassNames(data.classNames);
          setCrossValidated(data.crossValidated);
          setCvFoldAccuracies(data.cvFoldAccuracies);
          setCvStd(data.cvStd);
          setIsTraining(false);
          loadModel(data.model, data.dataset.maxLen, data.alphabet);
        }
      }, Math.max(15, Math.min(120, 3000 / fullHistory.length)));
    } catch (err) {
      setTrainError((err as Error).message);
      setIsTraining(false);
    }
  }

  async function loadModel(
    modelArtifacts: { modelTopology: unknown; weightSpecs: unknown; weightDataB64: string },
    maxLen: number,
    alphabet: string[]
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
    setModelAlphabet(alphabet);
  }

  return (
    <main className="min-h-screen bg-lab-bg text-lab-ink">
      <header className="border-b border-lab-border px-6 py-4">
        <h1 className="text-lg font-semibold">Sequence Classifier</h1>
        <p className="text-sm text-lab-dim">
          Works with DNA, RNA, protein, or any custom sequence alphabet — add examples per
          class, train a recurrent model, and test it live.
        </p>
      </header>

      {/* Encoding preview — kept standalone so you can inspect the encoding before training */}
      <div className="max-w-[1400px] mx-auto p-4 pb-0">
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">
            Encoding preview
          </h2>
          <SequenceTypeSelector config={seqType} onChange={setSeqType} />
          <textarea
            value={rawInput}
            onChange={(e) => setRawInput(e.target.value)}
            placeholder="Paste any sequence to preview its encoding…"
            className="w-full h-16 bg-lab-bg border border-lab-border rounded-md p-2 text-sm font-mono resize-none focus:outline-none focus:border-accent"
          />
          {validation.errors.length > 0 && (
            <p className="text-xs text-base-T">{validation.errors[0]}</p>
          )}
          {validation.valid && (
            <>
              <SequenceViewer sequence={validation.cleaned} alphabet={previewAlphabet} />
              <button
                onClick={() => setShowOneHot((v) => !v)}
                className="text-xs px-3 py-1.5 rounded-md bg-lab-panel2 border border-lab-border hover:border-accent transition-colors"
              >
                {showOneHot ? "Hide" : "Show"} one-hot encoding
              </button>
              {showOneHot && <OneHotViewer sequence={validation.cleaned} alphabet={previewAlphabet} />}
            </>
          )}
        </section>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-4 max-w-[1400px] mx-auto">
        {/* Column 1: Classes (Teachable-Machine style data) */}
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">Classes</h2>
          <ClassesPanel classes={classes} onChange={setClasses} />
        </section>

        {/* Column 2: Training */}
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">Training</h2>

          <ModelConfig hp={hp} onChange={setHp} disabled={isTraining} />

          <button
            onClick={handleTrain}
            disabled={isTraining || totalExamples < 10}
            className="w-full py-2 rounded-md bg-accent text-lab-bg font-medium text-sm disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90 transition-colors"
          >
            {isTraining ? "Training…" : "Train model"}
          </button>

          {trainError && <p className="text-xs text-base-T">{trainError}</p>}
          {warnings.length > 0 && (
            <div className="text-[11px] text-base-G space-y-0.5">
              {warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          <TrainingDashboard
            history={history}
            totalEpochs={hp.epochs}
            metrics={metrics}
            confusionMatrix={confusionMatrix}
            classNames={trainClassNames}
            isTraining={isTraining}
            crossValidated={crossValidated}
            cvFoldAccuracies={cvFoldAccuracies}
            cvStd={cvStd}
          />
        </section>

        {/* Column 3: Preview / inference */}
        <section className="bg-lab-panel border border-lab-border rounded-lg p-4 space-y-4">
          <h2 className="text-sm font-semibold text-lab-dim uppercase tracking-wide">Preview</h2>
          <InferencePanel
            model={model}
            maxLen={modelMaxLen}
            alphabet={modelAlphabet}
            classNames={trainClassNames}
          />
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
