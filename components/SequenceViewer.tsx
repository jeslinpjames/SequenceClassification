"use client";

const COLOR_CLASS: Record<string, string> = {
  A: "base-a",
  C: "base-c",
  G: "base-g",
  T: "base-t",
};

export default function SequenceViewer({ sequence }: { sequence: string }) {
  if (!sequence) return null;
  return (
    <div className="font-mono text-sm leading-relaxed break-all bg-lab-bg border border-lab-border rounded-md p-3 max-h-40 overflow-y-auto">
      {[...sequence].map((base, i) => (
        <span key={i} className={COLOR_CLASS[base] ?? "text-lab-dim"}>
          {base}
        </span>
      ))}
    </div>
  );
}
