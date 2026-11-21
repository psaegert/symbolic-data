// Length-conditioned term-pool prior: simulation library (reuses the sampler's verified T8 port).
const fs = require("fs");
const js = fs.readFileSync(__dirname + "/../shapes/main.js", "utf8");
const pure = js.split("// ------------------------------------------------------------------ colours and labels")[0]
  .replace(/^const \$ = id.*$/m, "").replace(/^try \{ const s = JSON.*$/m, "");
const M = new Function("localStorage", pure + "; return {mulberry, betaS, dirichlet, categorical, truncGeomP, T8N, sampleT8, countLeaves, pickLayer, DEFAULTS};")({ getItem: () => null });
const PHI = { w1: 0.5, gamma: 0.25, delta: 0.5, alphaFam: 0.36 };
const ABORT = { abort: true };
const sizePrior = nmax => { const w = []; for (let n = 0; n <= nmax; n++) w.push(Math.exp(Math.pow(n, 0.7))); const s = w.reduce((a, b) => a + b, 0); return w.map(x => x / s); };
const VARIANTS = {
  A: { name: "spec (all patterns p_T8, n<=17)", caps: [17, 17, 17, 17, 17] },
  B: { name: "deeper smaller (caps 17, 8, 4, 2, 1)", caps: [17, 8, 4, 2, 1] },
};
for (const v of Object.values(VARIANTS)) v.pn = v.caps.map(sizePrior);

// One draw of the spec with the variant's size priors; throws ABORT once more than nmax operators are placed.
function draw(V, r, nmax = 100) {
  const pL = M.truncGeomP(0.5, 5), pT = M.truncGeomP(0.5, 5);
  const Lam = 1 + M.categorical(pL, r), layers = [];
  for (let l = 1; l <= Lam; l++) {
    const T = l === 1 ? 1 : 1 + M.categorical(pT, r);
    const pRep = l === Lam ? 0 : M.betaS(2, 2, r);
    const w = M.dirichlet(T, 1, r), terms = [];
    for (let i = 0; i < T; i++) { const n = M.categorical(V.pn[l - 1], r); terms.push({ n, pattern: M.sampleT8(n, PHI, r) }); }
    layers.push({ T, pRep, w, terms });
  }
  let ops = 0;
  const copies = new Map();
  function inst(l, i) {
    const key = l + "." + i; copies.set(key, (copies.get(key) || 0) + 1);
    const term = layers[l - 1].terms[i];
    function clone(node) {
      if (node.k === "l") {
        if (r() < layers[l - 1].pRep) { const l2 = M.pickLayer(l, Lam, 0.1, r); return inst(l2, M.categorical(layers[l2 - 1].w, r)); }
        return { k: "l", o: key };
      }
      if (++ops > nmax) throw ABORT;
      const out = { k: node.k, op: node.op, o: key, c: node.c.map(clone) };
      if (node.e !== undefined) out.e = node.e;
      return out;
    }
    const top = clone(term.pattern); top.top = key; return top;
  }
  try { const root = inst(1, 0); return { root, copies, Lam, ops }; } catch (e) { if (e === ABORT) return null; throw e; }
}
function drawT8(r) { const n = M.categorical(M.T8N, r), root = M.sampleT8(n, PHI, r); root.top = "1.0"; return { root, copies: new Map([["1.0", 1]]), Lam: 1, ops: n }; }

// Per-expression statistics.
function stats(ex) {
  let ops = 0, leaves = 0, lits = 0, rootOps = 0;
  (function walk(n) { if (n.k === "l") { leaves++; return; } ops++; if (n.o === "1.0" || n.o === undefined) rootOps++; if (n.op === "pow" || n.op === "rootn") lits++; n.c.forEach(walk); })(ex.root);
  const summands = [];
  (function sum(n) { if (n.k === "b" && (n.op === "+" || n.op === "-")) { sum(n.c[0]); sum(n.c[1]); } else summands.push(n); })(ex.root);
  const topCount = new Map(); for (const s of summands) if (s.top && s !== ex.root) topCount.set(s.top, (topCount.get(s.top) || 0) + 1);
  const superpos = [...topCount.values()].some(v => v >= 2);
  let nCopies = 0, reuse = false, deepest = 1;
  for (const [k, v] of ex.copies) { nCopies += v; if (k !== "1.0" && v >= 2) reuse = true; deepest = Math.max(deepest, Number(k.split(".")[0])); }
  return { ops, tokens: ops + leaves + lits, summands: summands.length, superpos, nCopies, composite: nCopies >= 2, reuse, deepest, rootShare: rootOps / Math.max(1, ops) };
}

// Readable infix with empty leaves.
const PREC = { "+": 1, "-": 1, "*": 2, "/": 2 };
const SYM = { "+": " + ", "-": " − ", "*": "·", "/": "/" };
function fmtE(e) { return e < 0 ? `(${e})` : String(e); }
function infix(n) {
  if (n.k === "l") return "□";
  if (n.k === "b") {
    const p = PREC[n.op], a = n.c[0], b = n.c[1];
    const wrap = (c, right) => { const s = infix(c); if (c.k === "b" && (PREC[c.op] < p || (right && PREC[c.op] === p && (n.op === "-" || n.op === "/")))) return `(${s})`; return s; };
    return wrap(a, false) + SYM[n.op] + wrap(b, true);
  }
  const s = infix(n.c[0]), atom = n.c[0].k === "l" || (n.c[0].k === "u" && !["pow", "neg", "inv"].includes(n.c[0].op));
  if (n.op === "pow") return (atom ? s : `(${s})`) + "^" + fmtE(n.e);
  if (n.op === "rootn") return n.e === 2 ? `sqrt(${s})` : `root[${n.e}](${s})`;
  if (n.op === "neg") return "−" + (n.c[0].k === "b" ? `(${s})` : s);
  if (n.op === "inv") return `1/${atom ? s : `(${s})`}`;
  return `${n.op}(${s})`;
}

// Targets over N = 1..100 operators.
function target(kind) {
  const w = new Array(101).fill(0);
  for (let N = 1; N <= 100; N++) {
    if (kind === "t8ext") w[N] = N <= 50 ? Math.pow(N, 0.7) : Math.pow(50, 0.7) - (N - 50) / 10;
    else if (kind === "stretch") { const f = x => Math.pow(17 * x / 50, 0.7); w[N] = N <= 50 ? f(N) : f(50) - (N - 50) / 10; }
    else if (kind === "decay") w[N] = -N / 15;
  }
  const mx = Math.max(...w.slice(1)); const p = w.map((x, N) => N === 0 ? 0 : Math.exp(x - mx)); const s = p.reduce((a, b) => a + b, 0);
  return p.map(x => x / s);
}
module.exports = { M, PHI, VARIANTS, draw, drawT8, stats, infix, target };
