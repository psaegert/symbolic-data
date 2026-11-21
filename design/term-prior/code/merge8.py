"""Add the round-8 traced banks to the explorer's samples.json (keeps T8, round 6, round 7)."""
import json
old = [r for r in json.load(open("samples.json")) if not r["prior"].startswith("r8")]
new = json.load(open("bank_r8.json"))
bank = old + new
for k, r in enumerate(bank): r["id"] = k
with open("samples.json", "w") as f:
    f.write("[\n" + ",\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in bank) + "\n]\n")
import collections; print("bank", len(bank), dict(collections.Counter(r["prior"] for r in bank)))
