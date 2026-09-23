"use client";

import { colorForSymbol } from "@/lib/sequence";

export default function SequenceViewer({
  sequence,
  alphabet,
}: {
  sequence: string;
  alphabet: string[];
}) {
  if (!sequence) return null;
  return (
    <div className="font-mono text-sm leading-relaxed break-all bg-lab-bg border border-lab-border rounded-md p-3 max-h-40 overflow-y-auto">
      {[...sequence].map((symbol, i) => (
        <span key={i} style={{ color: colorForSymbol(symbol, alphabet) }}>
          {symbol}
        </span>
      ))}
    </div>
  );
}
