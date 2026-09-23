"use client";

import { SequenceTypeConfig, SequencePreset, PRESET_ALPHABETS, colorForSymbol } from "@/lib/sequence";

const PRESET_LABELS: Record<SequencePreset, string> = {
  dna: "DNA",
  rna: "RNA",
  protein: "Protein",
  auto: "Auto-detect",
  custom: "Custom",
};

function previewAlphabet(config: SequenceTypeConfig): string[] {
  if (config.preset === "dna") return [...PRESET_ALPHABETS.dna];
  if (config.preset === "rna") return [...PRESET_ALPHABETS.rna];
  if (config.preset === "protein") return [...PRESET_ALPHABETS.protein];
  if (config.preset === "custom") {
    return [...new Set((config.customAlphabet ?? "").toUpperCase().replace(/\s+/g, "").split(""))];
  }
  return []; // auto: resolved from data, nothing to preview yet
}

export default function SequenceTypeSelector({
  config,
  onChange,
}: {
  config: SequenceTypeConfig;
  onChange: (c: SequenceTypeConfig) => void;
}) {
  const preview = previewAlphabet(config);

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-5 gap-1">
        {(Object.keys(PRESET_LABELS) as SequencePreset[]).map((p) => (
          <button
            key={p}
            onClick={() => onChange({ ...config, preset: p })}
            className={`px-1.5 py-1.5 rounded-md text-[11px] border transition-colors ${
              config.preset === p
                ? "bg-accent/20 border-accent text-accent"
                : "bg-lab-panel2 border-lab-border text-lab-dim hover:border-lab-dim"
            }`}
          >
            {PRESET_LABELS[p]}
          </button>
        ))}
      </div>

      {config.preset === "custom" && (
        <input
          type="text"
          value={config.customAlphabet ?? ""}
          onChange={(e) => onChange({ ...config, customAlphabet: e.target.value })}
          placeholder="Type every symbol your sequences use, e.g. ACDEFGHIKLMNPQRSTVWY"
          className="w-full bg-lab-bg border border-lab-border rounded-md px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-accent"
        />
      )}

      {config.preset === "auto" ? (
        <p className="text-[11px] text-lab-dim">
          Alphabet will be detected from whatever symbols appear in your examples once you add
          them.
        </p>
      ) : (
        preview.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {preview.map((s) => (
              <span
                key={s}
                className="w-5 h-5 rounded-sm flex items-center justify-center text-[10px] font-mono text-lab-bg font-semibold"
                style={{ backgroundColor: colorForSymbol(s, preview) }}
              >
                {s}
              </span>
            ))}
          </div>
        )
      )}
    </div>
  );
}
