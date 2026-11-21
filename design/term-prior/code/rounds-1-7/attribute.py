import json, warnings, yaml, collections
warnings.filterwarnings("ignore")
from symbolic_data import LampleChartonCatalog
CFG = "/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"
cfg = yaml.safe_load(open(CFG)); pools = cfg["holdout_pools"]; cfg["holdout_pools"] = []
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine
rec = []; cur = {"pool": None}
orig = cat.holdout_family_prototype
def wrapped(tokens, assume_canonical=False, strict=False):
    p = orig(tokens, assume_canonical=assume_canonical, strict=strict)
    if strict:
        try: inf = eng.prefix_to_infix(list(tokens))
        except Exception: inf = " ".join(map(str, tokens))
        rec.append({"pool": cur["pool"], "law": inf, "proto": " ".join(p) if p else None})
    return p
cat.holdout_family_prototype = wrapped
for pool in pools:
    cur["pool"] = pool
    cat.register_holdout_pool(pool)
json.dump(rec, open("holdout_laws.json", "w"))
print(len(rec), "registered law renderings;", len(cat.holdout_manager.skeleton_hashes), "distinct prototypes")
c = collections.Counter(r["proto"] for r in rec)
print("most common prototypes:")
for p, n in c.most_common(25): print(f"  {n:5d}  {p}")
for key in ["x1", "+ x1 x1", "+ x1 + x1 x1", "/ x1 x1", "+ x1 x2", "+ x1 + x2 x3", "* x1 x2"]:
    ex = [f"{r['pool']}: {r['law']}" for r in rec if r["proto"] == key][:4]
    print(f"\nPROTO [{key}] from {c[key]} laws, e.g.:"); [print("   ", e) for e in ex]
