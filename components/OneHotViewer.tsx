"use client";

import { oneHotEncode, colorForSymbol } from "@/lib/sequence";

export default function OneHotViewer({
  sequence,
  alphabet,
}: {
  sequence: string;
  alphabet: string[];
}) {
  const matrix = oneHotEncode(sequence, alphabet, sequence.length);
  const cellSize = alphabet.length > 12 ? "w-4 h-4" : "w-5 h-5";
  const colWidth = alphabet.length > 12 ? "w-4" : "w-5";

  return (
    <div className="overflow-x-auto">
      <div className="inline-block">
        <div className="flex mb-1">
          <div className="w-8" />
          {[...sequence].map((symbol, i) => (
            <div key={i} className={`${colWidth} text-[10px] text-center text-lab-dim font-mono`}>
              {symbol}
            </div>
          ))}
        </div>
        {alphabet.map((symbol, rowIdx) => (
          <div key={symbol} className="flex items-center">
            <div className="w-8 text-xs font-mono text-lab-dim truncate">{symbol}</div>
            {matrix.map((row, colIdx) => (
              <div
                key={colIdx}
                className={`${cellSize} border border-lab-bg`}
                style={{
                  backgroundColor: row[rowIdx] === 1 ? colorForSymbol(symbol, alphabet) : "#171F25",
                }}
                title={`position ${colIdx + 1}: ${row[rowIdx]}`}
              />
            ))}
          </div>
        ))}
      </div>
      <p className="text-xs text-lab-dim mt-2">
        Each column is one position in the sequence; each row is one symbol in the alphabet (
        {alphabet.join(", ")}). Exactly one cell per column is lit — that&apos;s the one-hot
        vector for that position.
      </p>
    </div>
  );
}
