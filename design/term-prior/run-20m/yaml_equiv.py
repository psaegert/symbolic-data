import os, sys, copy, warnings, yaml
warnings.filterwarnings("ignore")
os.environ["POOL7_FN_WEIGHTS"] = "hyper=0.5"
WT = "/home/psaegert/Projects/symbolic-data/.claude/worktrees/term-prior"
sys.path.insert(0, WT + "/src")
import symbolic_data, numpy as np
from symbolic_data import build_catalog
from symbolic_data.term_prior import TermPriorCatalog
sys.path.insert(0, WT + "/design/term-prior/code"); os.chdir(WT + "/design/term-prior/code")
import pool7 as p
Y = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr-rl/runs/term_prior_20m/configs/v26.0-TP10c-20M/catalog_train.yaml"))
assert Y.pop("holdout_family") == "precise"; Y["holdout_pools"] = []
a = build_catalog(Y); b = TermPriorCatalog.from_config(dict(copy.deepcopy(p.cfg), term_prior={}))
assert type(a) is TermPriorCatalog
ta, tb = a._sampler(), b._sampler()
r1, r2 = np.random.default_rng(99), np.random.default_rng(99)
same = sum(ta.draw(r1)[0] == tb.draw(r2)[0] for _ in range(500))
print("YAML vs port config: identical", same, "of 500")
