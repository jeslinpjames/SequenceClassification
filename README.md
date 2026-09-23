# Sequence Classifier

A Teachable-Machine-style tool for biological sequences: add example sequences per class,
train a small RNN / LSTM / GRU / BiLSTM, and get instant predictions — all deployable as a
single Vercel project. Works with DNA, RNA, protein, or any custom single-character alphabet,
not just DNA.

## Data format

There's no CSV upload anymore — data entry mirrors Teachable Machine: each class is its own
card, and you paste or upload example sequences directly into it (one per line, or FASTA with
`>` headers). Add as many classes as you like (2 or more). Pick the sequence type above the
classes — DNA, RNA, Protein, Auto-detect (infers the alphabet from whatever characters appear
in your examples), or Custom (type your own alphabet, e.g. for non-biological categorical
sequences).

## How it's built (and why)

- **Frontend:** Next.js (App Router) + Tailwind, deployed as a normal Vercel site.
- **Training:** happens for real, on the server, in `app/api/train/route.ts`, using
  TensorFlow.js on Node. This is genuine backpropagation on your data — not a client-side
  simulation.
- **Inference:** after training, the trained model's weights are sent back to the browser
  once and loaded there with `tf.loadLayersModel`. Predictions after that run instantly in
  the browser with no extra server call.

  This split exists because **Vercel serverless functions are stateless** — a model trained
  during one request can't be "remembered" by a later request unless you add a database or
  file store. Handing the trained weights to the browser sidesteps that cleanly, at the cost
  of the model needing to be small enough to send over the wire (a few hundred KB to a few MB
  for the sizes this app supports — plenty for a demo classifier).

- **Time limit:** `app/api/train/route.ts` sets `maxDuration = 60`. Vercel's Hobby plan caps
  function duration lower than Pro by default — if training a larger dataset/epoch count times
  out, either reduce epochs/dataset size or use a Pro account and raise `maxDuration` in
  `vercel.json`.

## Getting a real dataset

A good, small, well-known dataset to start with: **UCI "Molecular Biology (Promoter Gene
Sequences)"** — 106 E. coli DNA sequences (57 bases each), labeled promoter vs. non-promoter.
Small enough to train in seconds.

Run `scripts/prepare_dataset.py` (see its header comment for the fetch snippet) to get a
`sequence,label` CSV, then split it into the two files the app wants pasted into its two
classes:

```python
import pandas as pd
df = pd.read_csv("promoters.csv")
df[df.label == 1].sequence.to_csv("promoter_examples.txt", index=False, header=False)
df[df.label == 0].sequence.to_csv("non_promoter_examples.txt", index=False, header=False)
```

Then open each `.txt` and paste its contents into the matching class's textarea (or use the
class card's "Upload" button to load the file directly — one sequence per line works as-is).

This same paste-or-upload flow works for any sequence type: protein families, RNA classes,
or a fully custom alphabet — just set the sequence type accordingly and put each category's
examples in its own class.

## Running locally

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Deploying to Vercel

1. Push this folder to a new GitHub repo.
2. Go to https://vercel.com/new and import that repo.
3. Framework preset: Next.js (auto-detected). No environment variables are required.
4. Deploy. That's it — the training API route ships as a serverless function automatically.

If you're on Vercel's Hobby plan and see training time out on larger datasets, either lower
epochs/dataset size in the UI, or upgrade to Pro and increase `maxDuration` in `vercel.json`.

## Project structure

```
app/
  page.tsx                 main 3-panel UI
  api/train/route.ts       real backend training endpoint
lib/
  dna.ts                   validation, one-hot encoding, CSV parsing
  modelBuilder.ts          RNN/LSTM/GRU/BiLSTM model construction
components/                UI panels (sequence viewer, one-hot grid, dataset upload,
                            hyperparameter controls, training charts, inference)
scripts/prepare_dataset.py fetches + converts the UCI promoter dataset
```
