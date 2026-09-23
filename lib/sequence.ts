// Generalized sequence utilities: no longer hard-coded to DNA. Any sequence made of
// single characters (DNA, RNA, protein, or a fully custom alphabet) works the same way.

export const PRESET_ALPHABETS = {
  dna: ["A", "C", "G", "T"],
  rna: ["A", "C", "G", "U"],
  protein: [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
  ],
} as const;

export type SequencePreset = "dna" | "rna" | "protein" | "auto" | "custom";

export interface SequenceTypeConfig {
  preset: SequencePreset;
  customAlphabet?: string; // used when preset === "custom", e.g. "ACDEFG..."
}

/** Strips whitespace and uppercases — the only universal cleanup we can safely assume. */
export function cleanSequence(raw: string): string {
  return raw.trim().toUpperCase().replace(/\s+/g, "");
}

/** Union of every distinct character across a set of sequences, sorted for stable ordering. */
export function detectAlphabet(sequences: string[]): string[] {
  const set = new Set<string>();
  for (const s of sequences) {
    for (const ch of cleanSequence(s)) set.add(ch);
  }
  return [...set].sort();
}

/** Resolves a chosen preset (or auto-detection) into the concrete list of symbols to encode. */
export function resolveAlphabet(config: SequenceTypeConfig, sequences: string[]): string[] {
  switch (config.preset) {
    case "dna":
      return [...PRESET_ALPHABETS.dna];
    case "rna":
      return [...PRESET_ALPHABETS.rna];
    case "protein":
      return [...PRESET_ALPHABETS.protein];
    case "custom": {
      const chars = cleanSequence(config.customAlphabet ?? "").split("");
      return [...new Set(chars)].sort();
    }
    case "auto":
    default:
      return detectAlphabet(sequences);
  }
}

export interface ValidationResult {
  valid: boolean;
  cleaned: string;
  errors: string[];
}

export function validateSequence(raw: string, alphabet: string[]): ValidationResult {
  const cleaned = cleanSequence(raw);
  const errors: string[] = [];

  if (cleaned.length === 0) {
    return { valid: false, cleaned, errors: ["Sequence is empty."] };
  }

  const alphabetSet = new Set(alphabet);
  const invalid = new Set([...cleaned].filter((c) => !alphabetSet.has(c)));
  if (invalid.size > 0) {
    errors.push(`Contains characters outside the chosen alphabet: ${[...invalid].join(", ")}`);
  }
  if (cleaned.length < 4) {
    errors.push("Very short sequence — a model can't learn much from it alone.");
  }

  return { valid: errors.length === 0, cleaned, errors };
}

/** One-hot encodes a sequence into [maxLen x alphabet.length], order = alphabet order. */
export function oneHotEncode(sequence: string, alphabet: string[], maxLen: number): number[][] {
  const seq = sequence.toUpperCase();
  const rows: number[][] = [];
  for (let i = 0; i < maxLen; i++) {
    const char = seq[i];
    const row = new Array(alphabet.length).fill(0);
    const idx = alphabet.indexOf(char);
    if (idx >= 0) row[idx] = 1;
    rows.push(row);
  }
  return rows;
}

export function flattenBatch(matrices: number[][][], channels: number): Float32Array {
  const n = matrices.length;
  const len = matrices[0]?.length ?? 0;
  const out = new Float32Array(n * len * channels);
  let p = 0;
  for (const m of matrices) {
    for (const row of m) {
      for (let c = 0; c < channels; c++) out[p++] = row[c];
    }
  }
  return out;
}

/** Splits pasted text into individual sequences: recognizes FASTA (">" headers) or plain one-per-line. */
export function parseSequenceList(text: string): string[] {
  const trimmed = text.trim();
  if (trimmed.length === 0) return [];

  if (trimmed.startsWith(">") || /\n>/.test(trimmed)) {
    // FASTA: each record is a header line followed by one or more sequence lines.
    return trimmed
      .split(/\n(?=>)/)
      .map((record) => {
        const lines = record.split("\n").filter((l) => l.trim().length > 0);
        return lines.slice(1).join("").trim(); // drop the ">header" line
      })
      .filter((s) => s.length > 0);
  }

  return trimmed
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
}

// Deterministic color per alphabet symbol so any alphabet (not just A/C/G/T) gets
// consistent, distinguishable colors in the viewer.
const PALETTE = [
  "#4ADE80", "#38BDF8", "#FBBF24", "#FB7185", "#A78BFA", "#F472B6",
  "#34D399", "#60A5FA", "#FCD34D", "#FB923C", "#2DD4BF", "#C084FC",
  "#F87171", "#A3E635", "#22D3EE", "#E879F9", "#FACC15", "#818CF8",
  "#4ADE80", "#38BDF8",
];

export function colorForSymbol(symbol: string, alphabet: string[]): string {
  const idx = alphabet.indexOf(symbol);
  if (idx === -1) return "#8A9AA5";
  return PALETTE[idx % PALETTE.length];
}
