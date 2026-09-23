# DNA Sequence Classifier

Paste a DNA sequence, see it one-hot encoded, train a small RNN / LSTM / GRU / BiLSTM
on your own labeled dataset, and get instant predictions — all deployable as a single
Vercel project.

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

The app expects a CSV with a `sequence` column and a `label` column (label can be `1`/`0`,
`+`/`-`, or `promoter`/`non-promoter`). A good, small, well-known dataset to start with:

**UCI "Molecular Biology (Promoter Gene Sequences)"** — 106 E. coli DNA sequences (57 bases
each), labeled promoter vs. non-promoter. Small enough to train in seconds, which matches
"small model is enough."

To fetch and convert it:

```bash
pip install ucimlrepo pandas
python scripts/prepare_dataset.py
```

This writes `data/promoters.csv`, ready to upload in the app's "Training dataset" panel.

If you outgrow 106 examples, larger public promoter datasets exist (e.g. the Genomic
Benchmarks project's human non-TATA promoter set, tens of thousands of sequences) — the app's
CSV format works the same way for those; just export sequence/label pairs to a CSV.

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
"# SequenceClassification" 
