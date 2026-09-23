"use client";

import { BASES, oneHotEncode } from "@/lib/dna";

const BASE_BG: Record<string, string> = {
  A: "bg-base-a",
  C: "bg-base-c",
  G: "bg-base-g",
  T: "bg-base-t",
};

export default function OneHotViewer({ sequence }: { sequence: string }) {
  const matrix = oneHotEncode(sequence, sequence.length);

  return (
    <div className="overflow-x-auto">
      <div className="inline-block">
        <div className="flex mb-1">
          <div className="w-6" />
          {[...sequence].map((base, i) => (
            <div
              key={i}
              className="w-5 text-[10px] text-center text-lab-dim font-mono"
            >
              {base}
            </div>
          ))}
        </div>
        {BASES.map((base, rowIdx) => (
          <div key={base} className="flex items-center">
            <div className="w-6 text-xs font-mono text-lab-dim">{base}</div>
            {matrix.map((row, colIdx) => (
              <div
                key={colIdx}
                className={`w-5 h-5 border border-lab-bg ${
                  row[rowIdx] === 1 ? BASE_BG[base] : "bg-lab-panel2"
                }`}
                title={`position ${colIdx + 1}: ${row[rowIdx]}`}
              />
            ))}
          </div>
        ))}
      </div>
      <p className="text-xs text-lab-dim mt-2">
        Each column is one base position; each row is a channel (A, C, G, T). Exactly
        one cell per column is lit — that&apos;s the one-hot vector for that base.
      </p>
    </div>
  );
}
