import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import yaml
from symbolic_data import LampleChartonCatalog
from simplipy.utils import explicit_constant_placeholders
import pool7 as p
old = [r for r in json.load(open("samples.json")) if r["prior"] in ("t8", "pool4")]
new = json.load(open("bank_r7.json")) + json.load(open("bank_r7c.json"))
cfg = yaml.safe_load(open(p.CFG)); cfg["holdout_family"] = "precise"
hcat = LampleChartonCatalog.from_config(cfg); heng = hcat.simplipy_engine
for r in new:
    s = r["simp"]
    try:
        _, c = explicit_constant_placeholders(heng.operators_to_realizations(s), inplace=False, convert_numbers_to_constant=False)
        r["held"] = bool(hcat.is_held_out(s, c, assume_canonical=hcat._skeleton_is_holdout_canonical))
    except Exception:
        r["held"] = True
    r["cls"] = p.record_classes(s)
bank = old + new
for k, r in enumerate(bank): r["id"] = k
with open("samples.json", "w") as f:
    f.write("[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in bank) + "\n]\n")
print("bank", len(bank), {k: sum(r["prior"] == k for r in bank) for k in ("t8", "pool4", "r7", "r7c")})
