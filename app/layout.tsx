import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DNA Sequence Classifier",
  description: "Paste DNA, one-hot encode it, train an RNN/LSTM/GRU/BiLSTM, and predict — in the browser.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  );
}
