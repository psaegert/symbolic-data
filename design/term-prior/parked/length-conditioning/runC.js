const lib = require("./lib.js"); const fs = require("fs");
const V = { caps: [17, 8, 4, 2, 1] }; V.pn = lib.VARIANTS.B.pn.slice(); V.pn[0] = new Array(18).fill(1 / 18);
const n = Number(process.argv[2]), r = lib.M.mulberry(44), rows = []; let aborted = 0;
for (let k = 0; k < n; k++) { const ex = lib.draw(V, r); if (!ex) { aborted++; continue; } const s = lib.stats(ex); rows.push([s.ops, s.tokens, s.summands, +s.superpos, s.nCopies, +s.reuse, s.deepest, +s.rootShare.toFixed(3)]); }
const d = JSON.parse(fs.readFileSync("rows.json")); d.C = { n, aborted, rows }; fs.writeFileSync("rows.json", JSON.stringify(d));
console.log("C over 100:", aborted / n);
