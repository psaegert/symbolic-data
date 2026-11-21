"""The term prior (research prototype, ``design/term-prior/``) as a generative catalog, ``type: term_prior``.

A skeleton is one draw of the round-10c term prior: a budgeted recursive term that is, with probability s (one
structure level per expression), a relation (repeat and combine, composition, pairing) and otherwise a Lample-Charton
tree whose slots hold atoms or copies of smaller terms. The trees, labels, literals, family urn and holdout come from
the wrapped Lample-Charton catalog, built from the same config. With the default parameters the sampler reproduces
the prototype ``design/term-prior/code/pool8.py`` (key ``r10c``) draw for draw under the same seed.

Config: a ``lample_charton`` catalog config plus a ``term_prior`` mapping (every key optional; defaults = round 10c).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from simplipy.utils import codify, explicit_constant_placeholders

from symbolic_data.config_io import load_config
from symbolic_data.errors import NoValidSampleFoundError
from symbolic_data.generative import LampleChartonCatalog, register_generative_catalog

# Round 10c (design/term-prior/README.md); every value can be overridden in the config's ``term_prior`` mapping.
DEFAULTS: dict[str, Any] = dict(
    flat=0.2, rho="beta21", alpha="logu", s_add=True, urn=True, lam=True, mutate=True, persist=True, B_scale=1.3,
    struct=True, s_fixed=True, relations=True, s_beta=(1.0, 1.0),
    rel_shares=(31.5, 12.0, 21.4, 13.9, 12.3, 8.9), rel_alpha=1.0,
    k_prod=([2, 3, 4, 5, 6], [0.51, 0.265, 0.13, 0.07, 0.025]), k_sum=([2, 3, 4, 5], [0.66, 0.205, 0.08, 0.055]),
    pow_wrap=0.33, div=0.26, coef=0.42, minus=0.24, p_exp=0.44,
    pair_prod=0.6, pair_div=0.4, pair_minus=0.25,
)
REL = ["repeat product", "repeat sum", "pairing", "affine", "power", "function"]


class TermPriorSampler:
    """The round-10c sampler bound to one Lample-Charton catalog (its tree sampler, engine, urn and literal priors)."""

    def __init__(self, cat: LampleChartonCatalog, params: dict[str, Any]):
        self.C = dict(DEFAULTS, **params)
        self.cat = cat
        self.eng = cat.simplipy_engine
        self.ar = dict(self.eng.operator_arity)
        self.vs = list(cat.variables)
        self.vset = set(self.vs)
        self.smp = cat.skeleton_sampler
        self.slot_ops = set(self.smp.typed_slots)
        self.NS = self.smp._nesting
        self.URN = self.NS["affinity"]
        self.UNARY = [o for o in self.smp.unary_operators if self.ar[o] == 1]
        self.BINARY = [o for o in self.smp.binary_operators if self.ar[o] == 2 and o not in self.slot_ops]
        # the budget's distribution is the catalog's operator-count distribution (T8: length_exponential)
        SS = cat.sample_strategy
        self.T8_N = np.arange(SS["min_operators"], SS["max_operators"] + 1)
        dist = SS.get("n_operator_distribution", "length_exponential")
        if dist == "length_exponential":
            p = np.exp(self.T8_N ** SS["power"] / SS["lambda"])
        elif dist == "length_proportional":
            p = self.T8_N.astype(float) ** SS["power"]
        elif dist == "equiprobable_lengths":
            p = np.ones(len(self.T8_N))
        else:
            raise ValueError(f"term prior: unsupported n_operator_distribution {dist!r}")
        self.T8_P = p / p.sum()
        self.KINDS = list(self.vs) + ["<c>"]
        self.MAX_KINDS = len(self.vs)
        self.PI_REL = np.array(self.C["rel_shares"], dtype=float) / 100
        self._persist: dict[str, Any] = {}

    # ---- pool7 helpers ------------------------------------------------------------------------------------------
    @staticmethod
    def num(v):
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else f"{f:.6g}"
        except Exception:
            return v

    def function_label(self, rng, counts):
        NS, URN = self.NS, self.URN
        if counts is None:
            return str(rng.choice(NS["F"], p=NS["pF"]))
        if URN["independent"] and rng.random() < URN["p_independent"]:
            return str(rng.choice(URN["independent"], p=URN["within_independent"]))
        p = URN["alpha"] * URN["pi"] + counts
        f = int(rng.choice(len(p), p=p / p.sum()))
        counts[f] += 1
        m = URN["members"][f]
        return m[0] if len(m) == 1 else str(rng.choice(m, p=URN["within"][f]))

    def q_bin(self, s_add):
        return np.array([s_add / 2 if op in ("+", "-") else (1 - s_add) / 2 for op in self.smp.binary_operators])

    def persist_table(self, nmax):
        if self._persist.get("nmax", -1) >= nmax:
            return self._persist["C"]
        smp, NS = self.smp, self.NS
        w1, w2, g, mF, mS = smp._n_unary_operators, smp._n_binary_operators, NS["gamma"], NS["mF"], NS["mS"]
        D = nmax + 2
        C = [[1.0] + [0.0] * nmax for _ in range(D + 1)]
        for m in range(1, nmax + 1):
            for c in range(D, -1, -1):
                binary = w2 * sum(C[c][i] * C[c][m - 1 - i] for i in range(m))
                C[c][m] = w1 * (mF * g ** c * C[min(c + 1, D)][m - 1] + mS * C[c][m - 1]) + binary
        self._persist.update(nmax=nmax, C=C)
        return C

    def sample_shape(self, n, rng, counts, q=None, c0=0, persist=False):
        smp, NS, ar = self.smp, self.NS, self.ar
        C = self.persist_table(max(n, 24)) if persist else smp._nesting_table(max(n, NS["nmax"]))
        D = len(C) - 1
        w1, w2, g, delta = smp._n_unary_operators, smp._n_binary_operators, NS["gamma"], NS["delta"]
        sticky = delta < 1.0
        out = []

        def emit(op, children):
            spec = smp.typed_slots.get(op)
            out.append(op)
            it = iter(children)
            for a in range(ar[op]):
                if spec is not None and a == spec["argument"]:
                    out.append(smp._slot_literal(op, rng))
                else:
                    grow(*next(it))

        def grow(c, m, k):
            if m == 0:
                out.append(None)
                return
            pF = w1 * NS["mF"] * g ** c * C[min(c + 1, D)][m - 1] if NS["F"] else 0.0
            pS = w1 * NS["mS"] * C[c][m - 1] if NS["S"] else 0.0
            cb = c if persist else 0
            splits = np.array([w2 * C[cb][i] * C[cb][m - 1 - i] for i in range(m)])
            pB = float(splits.sum())
            r = rng.random() * (pF + pS + pB)
            if r < pF:
                emit(self.function_label(rng, counts), [(min(c + 1, D), m - 1, None)])
            elif r < pF + pS:
                emit(str(rng.choice(NS["S"], p=NS["pS"])), [(c, m - 1, k)])
            else:
                i = int(rng.choice(m, p=splits / pB))
                base = smp.binary_operator_probs if q is None else q
                if sticky and k is not None:
                    pw = np.array([pb * (1.0 if kk == k else delta) for pb, kk in zip(base, NS["bin_class"])])
                    pw = pw if pw.sum() > 0 else np.ones_like(pw)
                    op = str(rng.choice(smp.binary_operators, p=pw / pw.sum()))
                else:
                    op = str(rng.choice(smp.binary_operators, p=base / base.sum()))
                k2 = NS["bin_class"][smp.binary_operators.index(op)] if sticky else None
                emit(op, [(cb, i, k2), (cb, m - 1 - i, k2)])
        grow(min(int(c0), D), int(n), None)
        return out

    def slot_depths(self, shape_t, persist):
        F = set(self.NS["F"])
        out = {}
        pos = 0
        ar, smp, slot_ops = self.ar, self.smp, self.slot_ops

        def walk(c):
            nonlocal pos
            t = shape_t[pos]
            pos += 1
            if t.startswith("@"):
                out[int(t[1:])] = c
                return
            a = ar.get(t, 0)
            for k in range(a):
                if t in slot_ops and k == smp.typed_slots[t]["argument"]:
                    pos += 1
                    continue
                if t in F:
                    walk(c + 1)
                elif a == 1:
                    walk(c)
                else:
                    walk(c if persist else 0)
        walk(0)
        return [out[i] for i in range(len(out))]

    def slot_mask(self, tokens):
        mask = [False] * len(tokens)
        pos = 0
        ar, smp, slot_ops = self.ar, self.smp, self.slot_ops

        def walk(is_slot):
            nonlocal pos
            i = pos
            t = tokens[pos]
            pos += 1
            mask[i] = is_slot
            a = ar.get(t, 0) if t is not None else 0
            for k in range(a):
                walk(t in slot_ops and k == smp.typed_slots[t]["argument"])
        walk(False)
        return mask

    def _slot_owner(self, tokens, i):
        pos = 0
        owner = {}
        ar, smp, slot_ops = self.ar, self.smp, self.slot_ops

        def walk():
            nonlocal pos
            t = tokens[pos]
            pos += 1
            for k in range(ar.get(t, 0)):
                if t in slot_ops and k == smp.typed_slots[t]["argument"]:
                    owner[pos] = t
                walk()
        walk()
        return owner[i]

    def modify_traced(self, tokens, rng, gate, P, counts):
        ar, smp, num = self.ar, self.smp, self.num
        out = list(tokens)
        ch = [None] * len(out)
        mask = self.slot_mask(out)
        family_of = {m: f for f, ms in zip(range(len(self.URN["members"])), self.URN["members"]) for m in ms} if self.URN else {}
        if rng.random() < P["p_struct"]:
            ops = [i for i, t in enumerate(out) if t in ar and t not in self.slot_ops]
            if ops:
                i = ops[int(rng.integers(len(ops)))]
                old = out[i]
                if ar[old] == 1:
                    new = self.function_label(rng, counts) if old in family_of else str(rng.choice(self.UNARY))
                    if ar.get(new, 1) != 1:
                        new = old
                else:
                    new = str(rng.choice(self.BINARY))
                if new != old:
                    out[i] = new
                    ch[i] = ["op", old]
        for i, t in enumerate(out):
            if t in ar:
                continue
            if mask[i]:
                if rng.random() < P["p_exp"]:
                    new = num(smp._slot_literal(self._slot_owner(out, i), rng))
                    if new != t:
                        out[i] = new
                        ch[i] = ["exp", t]
            elif t in self.vset:
                if gate and rng.random() < P["p_var_swap"]:
                    new = str(rng.choice(self.vs))
                    if new != t:
                        out[i] = new
                        ch[i] = ["var", t]
            else:
                if rng.random() < P["p_shared_const"]:
                    ch[i] = ["shared", t]
                else:
                    out[i] = num(smp._constant_literal(rng))
                    ch[i] = ["fresh", t]
        return out, ch

    def simplify_checked(self, raw):
        ar, vset, num = self.ar, self.vset, self.num
        had = any(x in vset for x in raw)
        try:
            s = [num(x) if (x not in ar and x not in vset) else x for x in self.eng.simplify(raw, mode=self.cat.simplify_mode)]
        except Exception:
            return None, "simplification failed"
        if had and not any(x in vset for x in s):
            return None, "every variable cancelled"
        if any("inf" in x or "nan" in x for x in s):
            return None, "non-finite literal"
        return s, None

    # ---- pool8 ---------------------------------------------------------------------------------------------------
    def root_prior(self):
        C = self.C
        pr = {
            "flat": lambda rng: bool(rng.random() < C["flat"]),
            "f": lambda rng: float(rng.beta(2, 3)),
            "rho": (lambda rng: float(rng.uniform())) if C["rho"] == "uniform" else (lambda rng: float(rng.beta(2, 1))),
            "alpha": (lambda rng: math.inf) if C["alpha"] is None else
                     (lambda rng: float(math.exp(rng.uniform(math.log(0.1), math.log(10))))),
        }
        if C["s_add"]:
            pr["s_add"] = lambda rng: float(rng.beta(0.5, 0.5))
        if C["mutate"]:
            pr["gate"] = lambda rng: bool(rng.random() < 0.5)
        if C.get("struct"):
            pr["s_struct"] = lambda rng: float(rng.beta(*C["s_beta"]))
        return pr

    @staticmethod
    def split_budget(rng, R, uses):
        k = len(uses)
        if R <= 0:
            return [0] * k, 0
        v = rng.dirichlet(np.ones(k)) if k > 1 else np.ones(1)
        bt = []
        for t in range(k):
            x = R * v[t] / uses[t]
            bt.append(int(math.floor(x)) + int(rng.random() < x - math.floor(x)))
        return bt, R - sum(bt[t] * uses[t] for t in range(k))

    def family_label(self, rng, counts):
        NS, URN = self.NS, self.URN
        if counts is None:
            F = [f for f in NS["F"] if f not in ("pow", "rootn")]
            w = np.array([NS["pF"][list(NS["F"]).index(f)] for f in F])
            return str(rng.choice(F, p=w / w.sum()))
        p = URN["alpha"] * URN["pi"] + counts
        f = int(rng.choice(len(p), p=p / p.sum()))
        counts[f] += 1
        m = URN["members"][f]
        return m[0] if len(m) == 1 else str(rng.choice(m, p=URN["within"][f]))

    # minimal prefix parse for the no-merge check (struct_scores / coverage of the prototype)
    _BIN = {"+", "-", "*", "/", "pow", "rootn"}

    def _parse(self, tokens):
        it = iter(tokens)

        def rec():
            t = next(it)
            a = 2 if t in self._BIN else 1 if t in self.ar and self.ar[t] == 1 else 0
            return (t, [rec() for _ in range(a)])
        return rec()

    def _is_const(self, node):
        t, kids = node
        return (not kids and t not in self.vset) or (bool(kids) and all(self._is_const(k) for k in kids))

    def _flatten(self, node, ops, through):
        t, kids = node
        if t in ops:
            return self._flatten(kids[0], ops, through) + self._flatten(kids[1], ops, through)
        if t in through:
            return self._flatten(kids[0], ops, through)
        return [node]

    def _operands(self, tokens, kind):
        node = self._parse(tokens)
        while node[0] == "neg":
            node = node[1][0]
        ops = self._flatten(node, {"+", "-"} if kind == "sum" else {"*", "/"}, {"neg"} if kind == "sum" else {"inv"})
        return [o for o in ops if not self._is_const(o)]

    @staticmethod
    def assemble(shape_t, leaves):
        raw, rawo = [], []
        for x in shape_t:
            if x.startswith("@"):
                lf = leaves[int(x[1:])]
                raw += lf["tokens"]
                rawo += [lf["pool"]] * len(lf["tokens"])
            else:
                raw.append(x)
                rawo.append(-1)
        return raw, rawo

    def relation_term(self, rng, b, depth, th, src, lam_d, alphabet, counts, lam, c0, fc, no_comp):
        C, NS, num, smp = self.C, self.NS, self.num, self.smp
        start = c0 if C["persist"] else 0
        w = C["rel_alpha"] * self.PI_REL + fc
        if no_comp:
            w[[REL.index("affine"), REL.index("power"), REL.index("function")]] = 0.0
        w[REL.index("function")] *= NS["gamma"] ** start
        if b < 2:
            w[REL.index("affine")] = 0.0
        j = int(rng.choice(len(REL), p=w / w.sum()))
        fc[j] += 1
        rel = REL[j]
        def child(bb, cc, nc=False): return self.term(rng, bb, depth + 1, th, alphabet, counts, lam, cc, True, fc, nc)
        pool, leaves, rounding = [], [], 0
        if rel.startswith("repeat"):
            kind = "product" if rel == "repeat product" else "sum"
            ks, ps = C["k_prod"] if kind == "product" else C["k_sum"]
            k = min(int(rng.choice(ks, p=np.array(ps) / sum(ps))), b + 1)
            R = b - (k - 1)
            wraps = []
            for i in range(k):
                p_wrap = C["pow_wrap"] if kind == "product" else C["coef"]
                if R >= 1 and rng.random() < p_wrap:
                    wraps.append(True)
                    R -= 1
                else:
                    wraps.append(False)
            bT = R // k
            rounding = R - k * bT
            tpl = child(bT, start)
            if kind == "product":
                ops = ["/" if rng.random() < C["div"] else "*" for _ in range(k - 1)]
            else:
                ops = ["-" if rng.random() < C["minus"] else "+" for _ in range(k - 1)]
            slot = []
            for i in range(k):
                if not wraps[i]:
                    slot.append([f"@{i}"])
                elif kind == "product":
                    op = str(rng.choice(["pow", "rootn"], p=[0.75, 0.25]))
                    slot.append([op, f"@{i}", num(smp._slot_literal(op, rng))])
                else:
                    slot.append(["*", num(smp._constant_literal(rng)), f"@{i}"])
            shape_t = list(reversed(ops)) + [t for sl in slot for t in sl]
            P = {"p_struct": 0.0, "p_exp": C["p_exp"], "p_var_swap": 0.7, "p_shared_const": 0.0}
            leaves = [{"pool": 0, "tokens": list(tpl["raw"]), "changes": [None] * len(tpl["raw"]), "how": "copy"}]
            for i in range(1, k):
                for attempt in range(5):
                    toks, ch = self.modify_traced(tpl["raw"], rng, True, dict(P, p_var_swap=0.7 if attempt == 0 else 1.0), counts)
                    trial = leaves + [{"pool": 0, "tokens": toks, "changes": ch, "how": "copy"}]
                    part, _ = self.assemble(list(reversed(ops[:i])) + [t for sl in slot[:i + 1] for t in sl], trial)
                    s_part, _ = self.simplify_checked(part)
                    if s_part is None or len(self._operands(s_part, kind)) >= i + 1:
                        break
                leaves = trial
            pool = [{"kind": "level", "tokens": tpl["raw"], "sub": tpl, "target": bT, "uses": k, "c0": start}]
            assign, label, n, budgets = [0] * k, f"{rel} of {k}", (k - 1) + sum(wraps), [bT]
        elif rel == "pairing":
            if rng.random() < C["pair_prod"]:
                op = "/" if rng.random() < C["pair_div"] else "*"
            else:
                op = "-" if rng.random() < C["pair_minus"] else "+"
            R = b - 1
            bA = int(round(rng.uniform() * R))
            bB = R - bA
            A, Bt = child(bA, start), child(bB, start)
            shape_t = [op, "@0", "@1"]
            pool = [{"kind": "level", "tokens": A["raw"], "sub": A, "target": bA, "uses": 1, "c0": start},
                    {"kind": "level", "tokens": Bt["raw"], "sub": Bt, "target": bB, "uses": 1, "c0": start}]
            leaves = [{"pool": t, "tokens": list(x["raw"]), "changes": [None] * len(x["raw"]), "how": "copy"} for t, x in enumerate((A, Bt))]
            assign, label, n, budgets = [0, 1], f"pairing {op}", 1, [bA, bB]
        else:
            if rel == "function":
                lab = self.family_label(rng, counts)
                shape_t = [lab, "@0"]
                own, inner_c = 1, start + 1
                label = f"function {lab}"
            elif rel == "power":
                op = str(rng.choice(["pow", "rootn"]))
                shape_t = [op, "@0", num(smp._slot_literal(op, rng))]
                own, inner_c = 1, start
                label = "power" if op == "pow" else "root"
            else:
                shape_t = ["+", "*", num(smp._constant_literal(rng)), "@0", num(smp._constant_literal(rng))]
                own, inner_c = 2, start
                label = "affine"
            A = child(b - own, inner_c, True)
            pool = [{"kind": "level", "tokens": A["raw"], "sub": A, "target": b - own, "uses": 1, "c0": inner_c}]
            leaves = [{"pool": 0, "tokens": list(A["raw"]), "changes": [None] * len(A["raw"]), "how": "copy"}]
            assign, n, budgets = [0], own, [b - own]
        raw, rawo = self.assemble(shape_t, leaves)
        H = {"depth": depth, "b": b, "n": n, "form": label, "relation": rel, "budgets": budgets, "rounding": rounding}
        return {"H": H, "shape": shape_t, "pool": pool, "assign": assign, "leaves": leaves, "raw": raw, "rawo": rawo}

    def term(self, rng, b, depth, parent, alphabet, counts, lam, c0, need_var, fc=None, no_comp=False):
        C, num, ar, vset, vs = self.C, self.num, self.ar, self.vset, self.vs
        lam_d = lam ** depth if parent is not None else 0.0
        th, src = {}, {}
        for key, draw in self.root_prior().items():
            if parent is not None and key == "s_struct" and C.get("s_fixed"):
                th[key], src[key] = parent[key], "expression"
            elif parent is not None and key in parent and rng.random() < lam_d:
                th[key], src[key] = parent[key], "inherited"
            else:
                th[key], src[key] = draw(rng), ("root" if parent is None else "redrawn")
        start = c0 if C["persist"] else 0
        if "s_struct" in th and b >= 1 and rng.random() < th["s_struct"]:
            if C.get("relations"):
                return self.relation_term(rng, b, depth, th, src, lam_d, alphabet, counts, lam, c0, fc, no_comp)
            raise NotImplementedError("round-9 operator-level forms are not ported (a recorded negative result)")
        n = b if th["flat"] else (max(1, min(b, int(round(th["f"] * b)))) if b >= 1 else 0)
        q = self.q_bin(th["s_add"]) if "s_add" in th else None
        shape = self.sample_shape(n, rng, counts, q, c0=start, persist=C["persist"])
        shape_t, L = [], 0
        for x in shape:
            if x is None:
                shape_t.append(f"@{L}")
                L += 1
            else:
                shape_t.append(num(x) if (x not in ar and x not in vset) else x)
        sd = [start + d for d in self.slot_depths(shape_t, C["persist"])] if n > 0 else [start]
        k = 1 + int(rng.binomial(L - 1, 1 - th["rho"])) if L > 1 else 1
        k = min(k, L, self.MAX_KINDS)
        if math.isinf(th["alpha"]):
            extra = [int(x) for x in rng.integers(0, k, L - k)]
        else:
            extra = [int(x) for x in rng.choice(k, L - k, p=rng.dirichlet([th["alpha"]] * k))]
        assign = list(range(k)) + extra
        rng.shuffle(assign)
        uses = [assign.count(t) for t in range(k)]
        R = b - n
        bt, unspent = self.split_budget(rng, R, uses)
        pool = [None] * k
        used, own_vars = set(), []

        def atom_kind():
            inherited = [x for x in alphabet if x not in used]
            if inherited and rng.random() < lam_d:
                x, how = str(rng.choice(inherited)), "parent's alphabet"
            else:
                fresh = [x for x in self.KINDS if x not in used] or self.KINDS
                x, how = str(rng.choice(fresh)), "fresh"
            used.add(x)
            return x, how
        atoms = [t for t in range(k) if bt[t] == 0]
        comps = [t for t in range(k) if bt[t] > 0]
        for t in atoms:
            x, how = atom_kind()
            pool[t] = {"kind": "const" if x == "<c>" else "var", "val": x, "uses": uses[t], "src": how}
            if x != "<c>":
                own_vars.append(x)
        if need_var and not comps and atoms and not own_vars:
            t = atoms[0]
            used.discard("<c>")
            cand = [x for x in alphabet if x not in used]
            x = str(rng.choice(cand)) if cand and rng.random() < lam_d else str(rng.choice([v for v in vs if v not in used]))
            pool[t] = {"kind": "var", "val": x, "uses": uses[t], "src": "forced: a term needs a variable"}
            own_vars.append(x)
        child_alpha = list(dict.fromkeys(list(alphabet) + own_vars))
        for t in comps:
            cj = max(sd[i] for i in range(L) if assign[i] == t) if C["persist"] else 0
            sub = self.term(rng, bt[t], depth + 1, th, child_alpha, counts, lam, cj, True, fc)
            pool[t] = {"kind": "level", "tokens": sub["raw"], "sub": sub, "target": bt[t], "uses": uses[t], "c0": cj}
        P = {"p_struct": 0.1 if C["mutate"] else 0.0, "p_exp": 0.5 if C["mutate"] else 0.0, "p_var_swap": 0.5, "p_shared_const": 0.0}
        gate = bool(th.get("gate", False))
        seen, leaves = set(), []
        for i in range(L):
            t = assign[i]
            pt = pool[t]
            if pt["kind"] == "var":
                leaves.append({"pool": t, "tokens": [pt["val"]], "how": "var"})
            elif pt["kind"] == "const":
                c = num(self.smp._constant_literal(rng))
                if t not in seen:
                    pt["val_first"] = c
                leaves.append({"pool": t, "tokens": [c], "how": "first" if t not in seen else "fresh"})
            elif t not in seen:
                leaves.append({"pool": t, "tokens": list(pt["tokens"]), "changes": [None] * len(pt["tokens"]), "how": "copy"})
            else:
                toks, ch = self.modify_traced(pt["tokens"], rng, gate, P, counts)
                leaves.append({"pool": t, "tokens": toks, "changes": ch, "how": "copy"})
            seen.add(t)
        for pt in pool:
            if pt["kind"] == "const":
                pt["val"] = pt.pop("val_first")
        raw, rawo = self.assemble(shape_t, leaves)
        H = {"depth": depth, "b": b, "n": n, "form": "LC tree", "theta": th, "budgets": bt, "rounding": unspent}
        return {"H": H, "shape": shape_t, "pool": pool, "assign": assign, "leaves": leaves, "raw": raw, "rawo": rawo}

    def draw(self, rng):
        """One accepted draw: (simplified tokens, raw tokens, budget, trace). Resamples cancellations, like the prototype."""
        C = self.C
        while True:
            B0 = int(rng.choice(self.T8_N, p=self.T8_P))
            B = int(round(B0 * C["B_scale"]))
            lam = float(rng.beta(3, 2)) if C["lam"] else 0.0
            counts = np.zeros(len(self.URN["members"])) if C["urn"] else None
            fc = np.zeros(len(REL))
            rec = self.term(rng, B, 0, None, [], counts, lam, 0, False, fc)
            s, _ = self.simplify_checked(rec["raw"])
            if s is None:
                continue
            return s, rec["raw"], B, rec


class TermPriorCatalog(LampleChartonCatalog):
    """A Lample-Charton catalog whose fresh skeletons come from the term prior (``type: term_prior``)."""

    term_prior_params: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "TermPriorCatalog":
        config_ = dict(load_config(config))
        params = dict(config_.pop("term_prior", None) or {})
        catalog = super().from_config(config_)
        catalog.term_prior_params = params
        catalog._term_sampler = None
        return catalog

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_term_sampler"] = None          # rebuilt on arrival from the catalog's own parts
        return state

    def _sampler(self) -> TermPriorSampler:
        if getattr(self, "_term_sampler", None) is None:
            self._term_sampler = TermPriorSampler(self, self.term_prior_params)
        return self._term_sampler

    def sample_skeleton(self, new: bool = False, decontaminate: bool = True, rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng()
        if len(self.skeletons) == 0 or new:
            tp = self._sampler()
            for _ in range(self.sample_strategy['max_tries']):
                skeleton, _, _, _ = tp.draw(rng)
                if tuple(skeleton) in self.skeletons or len(skeleton) > self.sample_strategy['max_length']:
                    continue
                executable = self.simplipy_engine.operators_to_realizations(skeleton)
                with_constants, constants = explicit_constant_placeholders(executable, inplace=True, convert_numbers_to_constant=False)
                try:
                    code_string = self.simplipy_engine.prefix_to_infix(with_constants, realization=True)
                except ValueError:
                    continue
                code = codify(code_string, self.variables + constants)
                if not decontaminate or not self.is_held_out(skeleton, constants, assume_canonical=self._skeleton_is_holdout_canonical):
                    return tuple(skeleton), code, constants
            raise NoValidSampleFoundError(f"Failed to sample a non-contaminated term-prior skeleton after {self.sample_strategy['max_tries']} retries")
        return super().sample_skeleton(new=new, decontaminate=decontaminate, rng=rng)


register_generative_catalog("term_prior", TermPriorCatalog)
