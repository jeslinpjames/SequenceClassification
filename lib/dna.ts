// Core DNA sequence utilities shared by client and server.

export const BASES = ["A", "C", "G", "T"] as const;
export type Base = (typeof BASES)[number];

export interface ValidationResult {
  valid: boolean;
  cleaned: string;
  errors: string[];
}

/**
 * Validates a raw pasted sequence: uppercases it, strips whitespace/newlines,
 * and confirms every character is one of A/C/G/T. Ambiguity codes (N, R, Y, ...)
 * are rejected here to keep the encoding simple for the demo models; this is
 * called out in the UI rather than silently coerced.
 */
export function validateSequence(raw: string): ValidationResult {
  const cleaned = raw.trim().toUpperCase().replace(/[\s\r\n]+/g, "");
  const errors: string[] = [];

  if (cleaned.length === 0) {
    errors.push("Sequence is empty.");
    return { valid: false, cleaned, errors };
  }

  const invalidChars = new Set(
    [...cleaned].filter((c) => !BASES.includes(c as Base))
  );
  if (invalidChars.size > 0) {
    errors.push(
      `Contains characters outside A/C/G/T: ${[...invalidChars].join(", ")}`
    );
  }
  if (cleaned.length < 8) {
    errors.push("Sequence is very short (< 8 bases) — encoding will still work, but a model can't learn much from it alone.");
  }

  return { valid: errors.length === 0, cleaned, errors };
}

/**
 * One-hot encodes a single sequence into a [length x 4] matrix, order A,C,G,T.
 * Pads with all-zero rows up to maxLen, or truncates if longer.
 */
export function oneHotEncode(sequence: string, maxLen: number): number[][] {
  const seq = sequence.toUpperCase();
  const rows: number[][] = [];
  for (let i = 0; i < maxLen; i++) {
    const char = seq[i];
    const row = [0, 0, 0, 0];
    const idx = BASES.indexOf(char as Base);
    if (idx >= 0) row[idx] = 1;
    rows.push(row);
  }
  return rows;
}

export function flattenBatch(matrices: number[][][]): Float32Array {
  const n = matrices.length;
  const len = matrices[0]?.length ?? 0;
  const out = new Float32Array(n * len * 4);
  let p = 0;
  for (const m of matrices) {
    for (const row of m) {
      out[p++] = row[0];
      out[p++] = row[1];
      out[p++] = row[2];
      out[p++] = row[3];
    }
  }
  return out;
}

export interface DatasetRow {
  sequence: string;
  label: number; // 0 or 1
}

export interface ParsedDataset {
  rows: DatasetRow[];
  errors: string[];
  maxLen: number;
}

/**
 * Parses a CSV with a header row containing "sequence" and "label" columns
 * (case-insensitive, order-independent). Label may be 0/1, true/false, or
 * +/- (the UCI promoter dataset convention).
 */
export function parseDatasetCSV(text: string): ParsedDataset {
  const errors: string[] = [];
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 2) {
    return { rows: [], errors: ["File needs a header row plus at least one data row."], maxLen: 0 };
  }

  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const seqIdx = header.findIndex((h) => h.includes("seq"));
  const labelIdx = header.findIndex((h) => h.includes("label") || h.includes("class"));

  if (seqIdx === -1 || labelIdx === -1) {
    return {
      rows: [],
      errors: ['Header must include a "sequence" column and a "label" (or "class") column.'],
      maxLen: 0,
    };
  }

  const rows: DatasetRow[] = [];
  let maxLen = 0;

  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length <= Math.max(seqIdx, labelIdx)) continue;

    const seqRaw = parts[seqIdx].trim();
    const { valid, cleaned } = validateSequence(seqRaw);
    if (!valid) {
      errors.push(`Row ${i + 1}: invalid sequence, skipped.`);
      continue;
    }

    const labelRaw = parts[labelIdx].trim().toLowerCase();
    let label: number | null = null;
    if (["1", "+", "true", "yes", "promoter"].includes(labelRaw)) label = 1;
    else if (["0", "-", "false", "no", "non-promoter", "nonpromoter"].includes(labelRaw)) label = 0;

    if (label === null) {
      errors.push(`Row ${i + 1}: unrecognized label "${parts[labelIdx]}", skipped.`);
      continue;
    }

    rows.push({ sequence: cleaned, label });
    maxLen = Math.max(maxLen, cleaned.length);
  }

  if (rows.length === 0) {
    errors.push("No valid rows found.");
  }

  return { rows, errors, maxLen };
}

export function classCounts(rows: DatasetRow[]): { positives: number; negatives: number } {
  let positives = 0;
  let negatives = 0;
  for (const r of rows) (r.label === 1 ? positives++ : negatives++);
  return { positives, negatives };
}
