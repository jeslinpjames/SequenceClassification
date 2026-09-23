"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  val_loss: number | null;
  val_acc: number | null;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-lab-panel2 border border-lab-border rounded-md px-3 py-2">
      <div className="text-[11px] text-lab-dim">{label}</div>
      <div className="text-lg font-mono text-accent">{value}</div>
    </div>
  );
}

export default function TrainingDashboard({
  history,
  totalEpochs,
  metrics,
  confusionMatrix,
  isTraining,
}: {
  history: EpochLog[];
  totalEpochs: number;
  metrics: Metrics | null;
  confusionMatrix: ConfusionMatrix | null;
  isTraining: boolean;
}) {
  const progress = totalEpochs > 0 ? history.length / totalEpochs : 0;

  return (
    <div className="space-y-4">
      {isTraining && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-lab-dim">
            <span>Training…</span>
            <span>
              epoch {history.length} / {totalEpochs}
            </span>
          </div>
          <div className="h-1.5 bg-lab-panel2 rounded-full overflow-hidden">
            <div
              className="h-full bg-accent transition-all duration-150"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="h-56 bg-lab-panel2 border border-lab-border rounded-md p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#232D34" />
              <XAxis dataKey="epoch" stroke="#8A9AA5" fontSize={11} />
              <YAxis stroke="#8A9AA5" fontSize={11} />
              <Tooltip
                contentStyle={{ background: "#171F25", border: "1px solid #232D34", fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="loss" stroke="#FB7185" dot={false} name="loss" />
              <Line type="monotone" dataKey="acc" stroke="#2DD4BF" dot={false} name="accuracy" />
              {history[0]?.val_loss !== null && (
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#FBBF24"
                  dot={false}
                  strokeDasharray="4 3"
                  name="val loss"
                />
              )}
              {history[0]?.val_acc !== null && (
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  stroke="#38BDF8"
                  dot={false}
                  strokeDasharray="4 3"
                  name="val accuracy"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {metrics && (
        <div className="grid grid-cols-4 gap-2">
          <MetricCard label="Val accuracy" value={`${(metrics.accuracy * 100).toFixed(1)}%`} />
          <MetricCard label="Precision" value={metrics.precision.toFixed(2)} />
          <MetricCard label="Recall" value={metrics.recall.toFixed(2)} />
          <MetricCard label="F1" value={metrics.f1.toFixed(2)} />
        </div>
      )}

      {confusionMatrix && (
        <div className="space-y-1">
          <div className="text-xs text-lab-dim">Confusion matrix (validation set)</div>
          <table className="text-xs font-mono border border-lab-border rounded-md overflow-hidden w-full">
            <thead>
              <tr className="bg-lab-panel2 text-lab-dim">
                <th className="p-1"></th>
                <th className="p-1">Pred +</th>
                <th className="p-1">Pred -</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-1 text-lab-dim bg-lab-panel2">Actual +</td>
                <td className="p-1 text-center bg-base-A/20">{confusionMatrix.tp}</td>
                <td className="p-1 text-center bg-base-T/10">{confusionMatrix.fn}</td>
              </tr>
              <tr>
                <td className="p-1 text-lab-dim bg-lab-panel2">Actual -</td>
                <td className="p-1 text-center bg-base-T/10">{confusionMatrix.fp}</td>
                <td className="p-1 text-center bg-base-A/20">{confusionMatrix.tn}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
