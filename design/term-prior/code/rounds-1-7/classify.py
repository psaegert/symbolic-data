"""Structural classes of prefix skeletons (canonical simplipy tokens)."""
ADD = {"+", "-"}

def parse(tokens, arity):
    it = iter(tokens)
    def rec():
        t = next(it)
        a = arity.get(t, 0)
        return (t, tuple(rec() for _ in range(a))) if a else t
    return rec()

def is_lit(t, varset):
    return isinstance(t, str) and t not in varset

def has_var(n, varset):
    if isinstance(n, str):
        return n in varset
    return any(has_var(c, varset) for c in n[1])

def num(t):
    try: return float(t)
    except Exception: return None

def is_poly(n, vs):
    if isinstance(n, str): return True
    op, ch = n
    if op in ("+", "-", "*"): return all(is_poly(c, vs) for c in ch)
    if op == "neg": return is_poly(ch[0], vs)
    if op == "pow":
        e = num(ch[1]) if isinstance(ch[1], str) else None
        return e is not None and e > 0 and float(e).is_integer() and is_poly(ch[0], vs)
    return False

def is_rational(n, vs):
    if isinstance(n, str): return True
    op, ch = n
    if op in ("+", "-", "*", "/"): return all(is_rational(c, vs) for c in ch)
    if op in ("neg", "inv"): return is_rational(ch[0], vs)
    if op == "pow":
        e = num(ch[1]) if isinstance(ch[1], str) else None
        return e is not None and float(e).is_integer() and is_rational(ch[0], vs)
    return False

def terms(n):
    """Flatten the top-level additive backbone (+, -, neg)."""
    if isinstance(n, str): return [n]
    op, ch = n
    if op in ADD: return terms(ch[0]) + terms(ch[1])
    if op == "neg": return terms(ch[0])
    return [n]

def factors(n):
    if isinstance(n, str): return [n]
    op, ch = n
    if op in ("*", "/"): return factors(ch[0]) + factors(ch[1])
    if op in ("neg", "inv"): return factors(ch[0])
    return [n]

def sig(n, vs):
    """Motif signature: literals -> C, variables -> V (identity dropped)."""
    if isinstance(n, str): return "V" if n in vs else "C"
    op, ch = n
    return "(" + op + " " + " ".join(sig(c, vs) for c in ch) + ")"

def motif(n, vs):
    """Signature of a term with its multiplicative literal coefficients removed."""
    fs = [f for f in factors(n) if has_var(f, vs)]
    return " ".join(sorted(sig(f, vs) for f in fs))

def classify(tokens, arity, vs):
    tree = parse(tokens, arity)
    nv = len({t for t in tokens if t in vs})
    poly = is_poly(tree, vs)
    rat = is_rational(tree, vs)
    T = [t for t in terms(tree) if has_var(t, vs)]
    k = len(T)
    motifs = [motif(t, vs) for t in T]
    from collections import Counter
    mc = Counter(motifs).most_common(1)[0][1] if motifs else 0
    proper_rat_terms = sum(1 for t in T if is_rational(t, vs) and not is_poly(t, vs))
    return {
        "nv": nv, "k_terms": k, "poly": poly, "rational": rat and not poly,
        "poly_ge3": poly and k >= 3,
        "poly_1v_ge3": poly and nv == 1 and k >= 3,
        "sum_rat_ge2": rat and proper_rat_terms >= 2,
        "affine_ge2v": poly and all(isinstance(t, str) or (t[0] in ("*",) and sum(has_var(c, vs) for c in t[1]) == 1 and any(isinstance(c, str) and c in vs for c in t[1])) for t in T) and nv >= 2 and k == nv,
        "repeat_motif_ge2": k >= 2 and mc >= 2,
        "superpos_ge3": k >= 3 and mc == k,
        "transcendental": any(t in TRANS for t in tokens),
    }

TRANS = {"sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","asinh","acosh","atanh","exp","log"}
