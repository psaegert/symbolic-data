"""Structure scores of one simplified prefix expression (no catalog needed).

repetition  share of operator nodes whose subtree, with variables and constants masked, occurs at least twice in the
            expression; + and - count as one operator, since a minus often only carries a constant's sign
            (x1/(x2 + 3) + x3/(x4 - 1) -> both quotients and both sums repeat).
alternation the most switches between the additive class (+ -) and the multiplicative class (* /) along any path from
            the root to a leaf; unary operators are looked through (a polynomial: 1, a rational function: 2).
top         the root's kind: sum, product, power, function, atom (neg and inv are looked through: -f is f's kind,
            1/f a product).
"""
BIN = {"+", "-", "*", "/", "pow", "rootn"}
UN = {"neg", "inv", "abs", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log"}
CLS = {"+": "A", "-": "A", "*": "M", "/": "M"}
FN = UN - {"neg", "inv"}

def parse(tokens):
    it = iter(tokens)
    def rec():
        t = next(it); a = 2 if t in BIN else 1 if t in UN else 0
        return (t, [rec() for _ in range(a)])
    return rec()

def is_var(t): return t.startswith("x") and t[1:].isdigit()

def masked(node, memo):
    t, kids = node
    if not kids: return "x" if is_var(t) else "c"
    if node[0] in ("pow", "rootn"):        # the exponent / index is structure: keep it
        s = f"{t}({masked(kids[0], memo)},{kids[1][0]})"
    else:
        s = f"{'+' if t == '-' else t}(" + ",".join(masked(k, memo) for k in kids) + ")"
    memo.append(s); return s

def repetition(tokens):
    memo = []; masked(parse(tokens), memo)
    if not memo: return 0.0
    from collections import Counter
    c = Counter(memo); return sum(1 for s in memo if c[s] >= 2) / len(memo)

def alternation(tokens):
    best = 0
    def walk(node, last, sw):
        nonlocal best
        t, kids = node
        cls = CLS.get(t)
        if cls is not None and last is not None and cls != last: sw += 1
        best = max(best, sw)
        for i, k in enumerate(kids):
            if t in ("pow", "rootn") and i == 1: continue
            walk(k, cls if cls is not None else last, sw)
    walk(parse(tokens), None, 0); return best

def top(tokens):
    t, kids = parse(tokens)
    while t == "neg": t, kids = kids[0]
    if not kids: return "atom"
    if t in ("+", "-"): return "sum"
    if t in ("*", "/", "inv"): return "product"
    if t in ("pow", "rootn"): return "power"
    return "function"

def scores(tokens):
    return {"rep": repetition(tokens), "alt": alternation(tokens), "top": top(tokens)}
