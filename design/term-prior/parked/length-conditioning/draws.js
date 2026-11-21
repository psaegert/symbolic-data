const lib = require("./lib.js"); const acc = require("./accept.json");
const C = { caps: [17, 8, 4, 2, 1] }; C.pn = lib.VARIANTS.B.pn.slice(); C.pn[0] = new Array(18).fill(1 / 18);
const VS = { A: lib.VARIANTS.A, C };
for (const t of ["stretch", "decay"]) for (const v of ["A", "C"]) {
  const r = lib.M.mulberry(2026 + (v === "C" ? 1 : 0) + (t === "decay" ? 10 : 0)), a = acc[`${v}_${t}`];
  console.log(`\n### ${v} | target ${t}`);
  let got = 0, tries = 0;
  while (got < 10) {
    tries++; const ex = lib.draw(VS[v], r); if (!ex) continue;
    if (r() >= a[ex.ops]) continue;
    got++; const s = lib.stats(ex);
    const cp = [...ex.copies].filter(([k]) => k !== "1.0").map(([k, n]) => `L${k.split(".")[0]}${"abcde"[+k.split(".")[1]]}×${n}`).join(" ");
    console.log(`${String(got).padStart(2)}. [${s.ops} ops${cp ? "; copies " + cp : ""}]  ${lib.infix(ex.root)}`);
  }
  console.log(`(${tries} proposals for 10 draws)`);
}
