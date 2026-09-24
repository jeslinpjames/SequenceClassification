"use client";

import { HyperParams, ModelType } from "@/lib/modelBuilder";

const MODEL_LABELS: Record<ModelType, string> = {
  rnn: "Simple RNN",
  lstm: "LSTM",
  gru: "GRU",
  bilstm: "Bidirectional LSTM",
};

function Field({
  label,
  value,
  children,
}: {
  label: string;
  value: string | number;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-lab-dim">
        <span>{label}</span>
        <span className="text-lab-ink font-mono">{value}</span>
      </div>
      {children}
    </div>
  );
}

export default function ModelConfig({
  hp,
  onChange,
  disabled,
}: {
  hp: HyperParams;
  onChange: (hp: HyperParams) => void;
  disabled?: boolean;
}) {
  const set = <K extends keyof HyperParams>(key: K, value: HyperParams[K]) =>
    onChange({ ...hp, [key]: value });

  return (
    <div className={`space-y-4 ${disabled ? "opacity-50 pointer-events-none" : ""}`}>
      <div className="grid grid-cols-4 gap-2">
        {(Object.keys(MODEL_LABELS) as ModelType[]).map((type) => (
          <button
            key={type}
            onClick={() => set("modelType", type)}
            className={`px-2 py-2 rounded-md text-xs border transition-colors ${
              hp.modelType === type
                ? "bg-accent/20 border-accent text-accent"
                : "bg-lab-panel2 border-lab-border text-lab-dim hover:border-lab-dim"
            }`}
          >
            {MODEL_LABELS[type]}
          </button>
        ))}
      </div>

      <Field label="Hidden units per layer" value={hp.hiddenUnits}>
        <input
          type="range"
          min={4}
          max={64}
          step={4}
          value={hp.hiddenUnits}
          onChange={(e) => set("hiddenUnits", Number(e.target.value))}
          className="w-full accent-accent"
        />
      </Field>

      <Field label="Layers" value={hp.numLayers}>
        <input
          type="range"
          min={1}
          max={3}
          step={1}
          value={hp.numLayers}
          onChange={(e) => set("numLayers", Number(e.target.value))}
          className="w-full accent-accent"
        />
      </Field>

      <Field label="Dropout" value={hp.dropout.toFixed(2)}>
        <input
          type="range"
          min={0}
          max={0.6}
          step={0.05}
          value={hp.dropout}
          onChange={(e) => set("dropout", Number(e.target.value))}
          className="w-full accent-accent"
        />
      </Field>

      <Field label="Learning rate" value={hp.learningRate}>
        <input
          type="range"
          min={0.001}
          max={0.05}
          step={0.001}
          value={hp.learningRate}
          onChange={(e) => set("learningRate", Number(e.target.value))}
          className="w-full accent-accent"
        />
      </Field>

      <div className="grid grid-cols-2 gap-4">
        <Field label="Batch size" value={hp.batchSize}>
          <input
            type="range"
            min={2}
            max={32}
            step={2}
            value={hp.batchSize}
            onChange={(e) => set("batchSize", Number(e.target.value))}
            className="w-full accent-accent"
          />
        </Field>
        <Field label="Epochs" value={hp.epochs}>
          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={hp.epochs}
            onChange={(e) => set("epochs", Number(e.target.value))}
            className="w-full accent-accent"
          />
        </Field>
      </div>

      <Field label="Validation split" value={`${Math.round(hp.valSplit * 100)}%`}>
        <input
          type="range"
          min={0.1}
          max={0.4}
          step={0.05}
          value={hp.valSplit}
          onChange={(e) => set("valSplit", Number(e.target.value))}
          className="w-full accent-accent disabled:opacity-40"
          disabled={hp.crossValidate}
        />
      </Field>

      <div className="border-t border-lab-border pt-3 space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs text-lab-dim flex items-center gap-2">
            <input
              type="checkbox"
              checked={hp.crossValidate}
              onChange={(e) => set("crossValidate", e.target.checked)}
              className="accent-accent"
            />
            k-fold cross-validation
          </label>
        </div>
        <p className="text-[11px] text-lab-dim -mt-2">
          Trains {hp.folds} models on different held-out slices and averages the result —
          slower, but a far more reliable number on a small dataset than one random split.
        </p>
        {hp.crossValidate && (
          <Field label="Folds" value={hp.folds}>
            <input
              type="range"
              min={3}
              max={10}
              step={1}
              value={hp.folds}
              onChange={(e) => set("folds", Number(e.target.value))}
              className="w-full accent-accent"
            />
          </Field>
        )}
      </div>

      <div className="border-t border-lab-border pt-3 space-y-1">
        <div className="flex justify-between items-center text-xs text-lab-dim">
          <span>Random seed</span>
          <button
            onClick={() => set("seed", Math.floor(Math.random() * 100000))}
            className="text-[11px] px-2 py-0.5 rounded bg-lab-panel2 border border-lab-border hover:border-accent transition-colors"
          >
            Randomize
          </button>
        </div>
        <input
          type="number"
          value={hp.seed}
          onChange={(e) => set("seed", Number(e.target.value))}
          className="w-full bg-lab-bg border border-lab-border rounded-md px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-accent"
        />
        <p className="text-[11px] text-lab-dim">
          Same seed + same data + same settings = same result every time. Change it to see how
          much a run varies by chance; keep it fixed to fairly compare model types.
        </p>
      </div>
    </div>
  );
}
