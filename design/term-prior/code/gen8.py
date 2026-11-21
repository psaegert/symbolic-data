"""Round-8 ladder draws: statistics rows for every step (plus T8 and round 7 as references) and traced banks for the
explorer. Usage: gen8.py N_STATS N_BANK  ->  r8_rows.json.gz, bank_r8.json"""
import sys, json, gzip, warnings, collections, multiprocessing as mp
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import numpy as np, yaml
from symbolic_data import LampleChartonCatalog
from simplipy.utils import explicit_constant_placeholders
import pool7 as p, pool8 as q

N, NB = int(sys.argv[1]), int(sys.argv[2])
KEYS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["t8", "r7"] + list(q.LADDER)
TAG = sys.argv[4] if len(sys.argv) > 4 else "r8"
cfg = yaml.safe_load(open(p.CFG)); cfg["holdout_family"] = "precise"
hcat = LampleChartonCatalog.from_config(cfg); heng = hcat.simplipy_engine
FSET = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log", "abs"}

def held(s):
    try:
        _, c = explicit_constant_placeholders(heng.operators_to_realizations(s), inplace=False, convert_numbers_to_constant=False)
        return bool(hcat.is_held_out(s, c, assume_canonical=hcat._skeleton_is_holdout_canonical))
    except Exception:
        return True

def fn_depth(t):
    pos, best = 0, 0
    def walk(c):
        nonlocal pos, best
        x = t[pos]; pos += 1; cc = c + (x in FSET); best = max(best, cc)
        for _ in range(p.ar.get(x, 0)): walk(cc)
    walk(0); return best

# Seed offsets of the first run (sorted step names), kept so the ladder reproduces those draws; r8 is new.
SEED = {"r7": 0, "r8a": 1, "r8b": 2, "r8c": 3, "r8d": 4, "r8e": 5, "r8f": 6, "t8": 7, "r8": 8, "r9": 9, "r9b": 10, "r10": 11, "r10b": 12, "r10c": 13}
HYP = {"sinh", "cosh", "tanh", "asinh", "acosh", "atanh"}
BINCLS = {"+": "A", "-": "A", "*": "M", "/": "M"}

def coherence(rec):
    """Round 8 traces only. var: [nested variable atoms whose variable an ancestor term already uses, all nested variable
    atoms]. cls: [nested trees whose majority binary class (+ - vs * /) equals the parent tree's, nested trees where
    both trees have a strict majority]."""
    var, cls = [0, 0], [0, 0]
    def majority(shape):
        a = sum(1 for t in shape if BINCLS.get(t) == "A"); m = sum(1 for t in shape if BINCLS.get(t) == "M")
        return "A" if a > m else "M" if m > a else None
    def walk(term, anc_vars, parent_major, depth):
        major = majority(term["shape"])
        if depth > 0:
            for p_ in term["pool"]:
                if p_["kind"] == "var":
                    var[1] += 1; var[0] += p_["val"] in anc_vars
            if major is not None and parent_major is not None:
                cls[1] += 1; cls[0] += major == parent_major
        own = {p_["val"] for p_ in term["pool"] if p_["kind"] == "var"}
        for p_ in term["pool"]:
            if p_["kind"] == "level": walk(p_["sub"], anc_vars | own, major, depth + 1)
    walk(rec, set(), None, 0)
    return {"var": var, "cls": cls}

def stat_row(r, key):
    s = r["simp"]
    top = [x["uses"] for x in r.get("pool", []) if x["kind"] in ("term", "level")] if not r.get("H", {}).get("const_law") else []
    B = r.get("B", r.get("H", {}).get("n_ops"))
    return {"simp": s, "raw_ops": sum(1 for x in r["raw"] if x in p.ar), "B": B, "levels": r.get("levels", 1),
            "uses": top, "cls": p.record_classes(s), "held": held(s), "fn_depth": fn_depth(s),
            "nfn": sum(1 for x in s if x in FSET), "fn_raw": [x for x in r["raw"] if x in FSET],
            "coh": coherence(r) if key[:2] in ("r8", "r9") or key.startswith("r10") else None,
            "form": r["H"].get("form") if key[:2] in ("r8", "r9") or key.startswith("r10") else None}

def run(key):
    rng = np.random.default_rng(20261122 + SEED[key])
    rows, bank = [], []
    for i in range(N):
        if key == "t8": r = p.draw_t8(rng)
        elif key == "r7":
            r = p.draw_r7(rng, dict(p.P7, B_scale=1.3))
            r["levels"] = (p.levels_used(r) + 1) if not r["H"].get("const_law") else 1
        else: r = q.draw_r8(rng, q.LADDER[key])
        row = stat_row(r, key); rows.append(row)
        if (key[:2] in ("r8", "r9") or key.startswith("r10")) and i < NB:
            r["prior"] = key; r["held"] = row["held"]; r["cls"] = row["cls"]; r["coh"] = row["coh"]; bank.append(r)
    return key, rows, bank

if __name__ == "__main__":
    with mp.get_context("fork").Pool(min(4, len(KEYS))) as pool:
        out = pool.map(run, KEYS)
    json.dump({k: rows for k, rows, _ in out}, gzip.open(f"rows_{TAG}.json.gz", "wt"))
    json.dump([r for _, _, b in out for r in b], open(f"bank_{TAG}.json", "w"))
    print("done", {k: len(rows) for k, rows, _ in out})
