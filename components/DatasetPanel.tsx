"use client";

import { useRef, useState } from "react";
import { DatasetRow, parseDatasetCSV, classCounts } from "@/lib/dna";

export default function DatasetPanel({
  onDataset,
}: {
  onDataset: (rows: DatasetRow[]) => void;
}) {
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [fileName, setFileName] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  function handleText(text: string, name: string) {
    const parsed = parseDatasetCSV(text);
    setRows(parsed.rows);
    setErrors(parsed.errors.slice(0, 5));
    setFileName(name);
    onDataset(parsed.rows);
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => handleText(String(reader.result), file.name);
    reader.readAsText(file);
  }

  const counts = classCounts(rows);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <button
          onClick={() => inputRef.current?.click()}
          className="px-3 py-1.5 rounded-md bg-lab-panel2 border border-lab-border text-sm hover:border-accent transition-colors"
        >
          Upload CSV
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.txt"
          onChange={handleFile}
          className="hidden"
        />
        <span className="text-xs text-lab-dim truncate">
          {fileName || "columns: sequence, label (0/1, +/-, promoter/non-promoter)"}
        </span>
      </div>

      {errors.length > 0 && (
        <div className="text-xs text-base-T bg-base-T/10 border border-base-T/30 rounded p-2 space-y-0.5">
          {errors.map((e, i) => (
            <div key={i}>{e}</div>
          ))}
        </div>
      )}

      {rows.length > 0 && (
        <div className="text-sm space-y-2">
          <div className="flex gap-4 text-lab-dim">
            <span>
              <span className="text-lab-ink font-medium">{rows.length}</span> sequences
            </span>
            <span>
              <span className="base-a font-medium">{counts.positives}</span> promoter
            </span>
            <span>
              <span className="base-t font-medium">{counts.negatives}</span> non-promoter
            </span>
          </div>
          <div className="max-h-32 overflow-y-auto border border-lab-border rounded-md">
            <table className="w-full text-xs font-mono">
              <tbody>
                {rows.slice(0, 8).map((r, i) => (
                  <tr key={i} className="border-b border-lab-border last:border-0">
                    <td className="px-2 py-1 truncate max-w-[220px]">{r.sequence}</td>
                    <td className="px-2 py-1 text-right text-lab-dim">
                      {r.label === 1 ? "promoter" : "non-promoter"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
