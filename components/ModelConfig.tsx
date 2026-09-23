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

      <Field label="Recurrent layers" value={hp.numLayers}>
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
          className="w-full accent-accent"
        />
      </Field>
    </div>
  );
}
