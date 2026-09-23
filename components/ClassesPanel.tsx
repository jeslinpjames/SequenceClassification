"use client";

import { useRef, useState } from "react";
import { parseSequenceList, parseClassesFromCSV } from "@/lib/sequence";

export interface SeqClass {
  id: string;
  name: string;
  sequences: string[];
}

function newClass(n: number): SeqClass {
  return { id: crypto.randomUUID(), name: `Class ${n}`, sequences: [] };
}

function ClassCard({
  cls,
  onUpdate,
  onRemove,
  removable,
}: {
  cls: SeqClass;
  onUpdate: (next: SeqClass) => void;
  onRemove: () => void;
  removable: boolean;
}) {
  const [draft, setDraft] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  function addFromDraft() {
    const parsed = parseSequenceList(draft);
    if (parsed.length === 0) return;
    onUpdate({ ...cls, sequences: [...cls.sequences, ...parsed] });
    setDraft("");
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = parseSequenceList(String(reader.result));
      onUpdate({ ...cls, sequences: [...cls.sequences, ...parsed] });
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  function removeExample(idx: number) {
    onUpdate({ ...cls, sequences: cls.sequences.filter((_, i) => i !== idx) });
  }

  return (
    <div className="border border-lab-border rounded-md p-3 space-y-2 bg-lab-panel2">
      <div className="flex items-center gap-2">
        <input
          value={cls.name}
          onChange={(e) => onUpdate({ ...cls, name: e.target.value })}
          className="flex-1 bg-transparent text-sm font-medium border-b border-transparent hover:border-lab-border focus:border-accent focus:outline-none px-0.5"
        />
        <span className="text-[11px] text-lab-dim whitespace-nowrap">
          {cls.sequences.length} example{cls.sequences.length !== 1 ? "s" : ""}
        </span>
        {removable && (
          <button
            onClick={onRemove}
            className="text-lab-dim hover:text-base-T text-xs px-1"
            title="Remove class"
          >
            ✕
          </button>
        )}
      </div>

      {cls.sequences.length > 0 && (
        <div className="max-h-24 overflow-y-auto space-y-1">
          {cls.sequences.map((s, i) => (
            <div
              key={i}
              className="flex items-center justify-between gap-2 bg-lab-bg rounded px-2 py-1"
            >
              <span className="font-mono text-[11px] truncate">{s}</span>
              <button
                onClick={() => removeExample(i)}
                className="text-lab-dim hover:text-base-T text-[11px] shrink-0"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="Paste sequences — one per line, or FASTA (>header + sequence)…"
        className="w-full h-16 bg-lab-bg border border-lab-border rounded-md p-2 text-xs font-mono resize-none focus:outline-none focus:border-accent"
      />
      <div className="flex gap-2">
        <button
          onClick={addFromDraft}
          disabled={draft.trim().length === 0}
          className="flex-1 py-1.5 rounded-md bg-lab-bg border border-lab-border text-xs hover:border-accent disabled:opacity-40 transition-colors"
        >
          Add examples
        </button>
        <button
          onClick={() => fileRef.current?.click()}
          className="px-2 py-1.5 rounded-md bg-lab-bg border border-lab-border text-xs hover:border-accent transition-colors"
        >
          Upload
        </button>
        <input ref={fileRef} type="file" accept=".txt,.fasta,.fa,.csv" onChange={handleFile} className="hidden" />
      </div>
    </div>
  );
}

export default function ClassesPanel({
  classes,
  onChange,
}: {
  classes: SeqClass[];
  onChange: (classes: SeqClass[]) => void;
}) {
  const [csvStatus, setCsvStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const csvFileRef = useRef<HTMLInputElement>(null);

  function updateClass(id: string, next: SeqClass) {
    onChange(classes.map((c) => (c.id === id ? next : c)));
  }

  function removeClass(id: string) {
    onChange(classes.filter((c) => c.id !== id));
  }

  function addClass() {
    onChange([...classes, newClass(classes.length + 1)]);
  }

  function handleCsvFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const { classes: parsed, errors } = parseClassesFromCSV(String(reader.result));
      if (parsed.length < 2) {
        setCsvStatus({ ok: false, message: errors[0] || "Couldn't parse this CSV." });
        return;
      }
      onChange(
        parsed.map((c) => ({ id: crypto.randomUUID(), name: c.name, sequences: c.sequences }))
      );
      const total = parsed.reduce((s, c) => s + c.sequences.length, 0);
      setCsvStatus({
        ok: true,
        message: `Loaded ${total} sequences across ${parsed.length} classes (${parsed
          .map((c) => c.name)
          .join(", ")}). This replaced your current classes.`,
      });
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <button
          onClick={() => csvFileRef.current?.click()}
          className="text-xs px-3 py-1.5 rounded-md bg-lab-panel2 border border-lab-border hover:border-accent transition-colors"
        >
          Upload CSV
        </button>
        <span className="text-[11px] text-lab-dim">
          columns: sequence, label — builds classes from the distinct label values
        </span>
        <input
          ref={csvFileRef}
          type="file"
          accept=".csv,.txt"
          onChange={handleCsvFile}
          className="hidden"
        />
      </div>
      {csvStatus && (
        <p className={`text-[11px] ${csvStatus.ok ? "text-lab-dim" : "text-base-T"}`}>
          {csvStatus.message}
        </p>
      )}

      {classes.map((cls) => (
        <ClassCard
          key={cls.id}
          cls={cls}
          onUpdate={(next) => updateClass(cls.id, next)}
          onRemove={() => removeClass(cls.id)}
          removable={classes.length > 2}
        />
      ))}
      <button
        onClick={addClass}
        className="w-full py-2 rounded-md border border-dashed border-lab-border text-sm text-lab-dim hover:border-accent hover:text-accent transition-colors"
      >
        + Add a class
      </button>
    </div>
  );
}

export function makeInitialClasses(): SeqClass[] {
  return [newClass(1), newClass(2)];
}
