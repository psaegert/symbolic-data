const { M, VARIANTS, draw, drawT8, stats, target } = require("./lib.js");
const fs = require("fs");
const n = Number(process.argv[2] || 200000), out = {};
const t0 = Date.now();
for (const [key, V] of Object.entries(VARIANTS)) {
  const r = M.mulberry(key === "A" ? 11 : 22);
  const rows = []; let aborted = 0;
  for (let k = 0; k < n; k++) { const ex = draw(V, r); if (!ex) { aborted++; continue; } const s = stats(ex); rows.push([s.ops, s.tokens, s.summands, +s.superpos, s.nCopies, +s.reuse, s.deepest, +s.rootShare.toFixed(3)]); }
  out[key] = { n, aborted, rows };
  console.log(key, "draws", n, "over 100:", (aborted / n).toFixed(4), "time", (Date.now() - t0) / 1000);
}
{ const r = M.mulberry(33), rows = []; for (let k = 0; k < 200000; k++) { const s = stats(drawT8(r)); rows.push([s.ops, s.tokens, s.summands, 0, 1, 0, 1, 1]); } out.T8 = { n: 200000, aborted: 0, rows }; }
fs.writeFileSync("rows.json", JSON.stringify(out));
