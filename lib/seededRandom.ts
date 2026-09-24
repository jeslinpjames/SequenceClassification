// Small deterministic PRNG (mulberry32) so "same seed" really means "same result" —
// Math.random() can't be seeded, which is why results were unreproducible before.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seededShuffle<T>(arr: T[], rng: () => number): T[] {
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/**
 * Splits row indices into `folds` groups, stratified by label so every fold has
 * roughly the same class balance as the full dataset — critical on small datasets
 * where a plain random split can easily be lopsided.
 */
export function stratifiedFolds(labels: number[], folds: number, rng: () => number): number[][] {
  const byLabel = new Map<number, number[]>();
  labels.forEach((label, idx) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label)!.push(idx);
  });

  const foldBuckets: number[][] = Array.from({ length: folds }, () => []);
  for (const indices of byLabel.values()) {
    const shuffled = seededShuffle(indices, rng);
    shuffled.forEach((idx, i) => foldBuckets[i % folds].push(idx));
  }
  return foldBuckets;
}
