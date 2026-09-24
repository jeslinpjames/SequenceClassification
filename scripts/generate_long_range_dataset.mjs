function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const BASES = ["A", "C", "G", "T"];

function generateExample(rng, seqLen, markerLen) {
  const bit0 = rng() < 0.5 ? 0 : 1;
  const bit1 = rng() < 0.5 ? 0 : 1;
  const startMarker = (bit0 === 0 ? "A" : "G").repeat(markerLen);
  const endMarker = (bit1 === 0 ? "C" : "T").repeat(markerLen);
  const fillerLen = seqLen - 2 * markerLen;
  let filler = "";
  for (let i = 0; i < fillerLen; i++) filler += BASES[Math.floor(rng() * 4)];
  const label = bit0 ^ bit1; // 0 = "match", 1 = "mismatch"
  return { sequence: startMarker + filler + endMarker, label };
}

const seqLen = 34, markerLen = 4, n = 300, seed = 7;
const rng = mulberry32(seed);
const rows = Array.from({ length: n }, () => generateExample(rng, seqLen, markerLen));

const lines = ["sequence,label"];
for (const r of rows) lines.push(`${r.sequence},${r.label === 0 ? "match" : "mismatch"}`);

const fs = await import("fs");
fs.writeFileSync("/mnt/user-data/outputs/long_range_dependency.csv", lines.join("\n") + "\n");
console.log(`Wrote ${rows.length} rows.`);
console.log("class balance:", rows.filter(r=>r.label===0).length, "match /", rows.filter(r=>r.label===1).length, "mismatch");
console.log("example rows:");
console.log(lines[1]);
console.log(lines[2]);
