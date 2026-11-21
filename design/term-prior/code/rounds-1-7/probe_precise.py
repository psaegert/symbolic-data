import sys, warnings, yaml
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/psaegert/Projects/symbolic-data/.claude/worktrees/holdout-precise/src")
import symbolic_data
from symbolic_data import LampleChartonCatalog
from simplipy.utils import explicit_constant_placeholders
assert "holdout-precise" in symbolic_data.__file__
cfg = yaml.safe_load(open("/home/psaegert/Projects/flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml"))
mode = sys.argv[1]; pools = sys.argv[2].split(",") if len(sys.argv) > 2 else cfg["holdout_pools"]
cfg["holdout_pools"] = pools; cfg["holdout_family"] = mode
cat = LampleChartonCatalog.from_config(cfg)
eng = cat.simplipy_engine
print(mode, "prototypes:", len(cat.holdout_manager.skeleton_hashes))
probes = ["x1", "2.3*x1 + 0.7", "1.3*x1**2 + 0.4", "1.3*x1**3 + 0.4", "1.3*x1**2 + 0.4*x1 + 2.1", "1.3*x1**4 + 0.4*x1 + 2.1",
 "1.3*x1**3 + 0.4*x1**2 - 0.8*x1 + 2.1", "1.3*x1**4 + 0.4*x1**2 - 0.8*x1 + 2.1", "0.5*x1**5 - 1.3*x1**3 + 0.4*x1 + 2.1",
 "1.3*x1*x2 + 0.5*x1 + 0.2", "1.3*x1**2 + 0.6*x2**2 + 0.2", "0.3*x1 + 0.7*x2 + 1.1*x3 + 0.2",
 "1.5/(x1 + 0.3) + 2.2/(x1 - 1.7)", "1.5/(x1 + 0.3) + 2.2/(x2 - 1.7)", "1.5/(x1**2 + 0.3) + 2.2",
 "(x1 + 0.4)/(x1**2 + 1.3)", "2.1*exp(-1.3*x1) + 0.7*exp(-0.2*x1)", "1.2*sin(0.7*x1) + 0.4*sin(2.3*x1)",
 "1.2*exp(-0.3*x1)", "2**x1", "x1*exp(-0.5*x2)", "1.7*x1*x2/x3**2", "1.3*x1**2*x2 + 0.4*x3",
 "1 + 1.2*x1**2/(x1**2 - 0.3) + 0.4*x1**2/(x1**2 - 2.5)"]
for s in probes:
    sk = ["pow" if t in ("**", "^") else t for t in eng.infix_to_prefix(s)]
    sk = list(eng.simplify(sk, mode=cat.simplify_mode))
    _, consts = explicit_constant_placeholders(eng.operators_to_realizations(sk), inplace=False, convert_numbers_to_constant=False)
    proto = cat.holdout_family_prototype(sk)
    ho = cat.is_held_out(sk, consts)
    print(f"{'HELD ' if ho else 'train'}  {s:58s} proto={' '.join(proto) if proto else proto}", flush=True)
