"""symbolic_data -- the model-agnostic symbolic-regression data layer.

Skeleton/expression sampling, priors, (X, y) support sampling, holdout management, and
dataset construction -- carved out of flash-ansr so symbolic-regression methods and the
srbf eval framework share one data substrate. Depends only on simplipy + numpy/sklearn.
"""
from symbolic_data.skeleton_pool import SkeletonPool, NoValidSampleFoundError
from symbolic_data.holdout import HoldoutManager
from symbolic_data.skeleton_sampling import SkeletonSampler
from symbolic_data.support_sampling import SupportSampler, SupportSamplingError
from symbolic_data.distributions import get_distribution, fastsrb_dist, DISTRIBUTIONS, BASE_DISTRIBUTIONS
from symbolic_data.prior_factory import build_prior_callable
from symbolic_data.registry import Registry
from symbolic_data.problem import Problem
from symbolic_data.samples import Sample, sample_from_skeleton, iter_samples
from symbolic_data.tensor_ops import mask_unused_variable_columns
from symbolic_data.datasets import load_benchmark, load_spec, BENCHMARKS
from symbolic_data.benchmarks import SpecBenchmark, FastSRBBenchmark
from symbolic_data.convert_data import ParserFactory, TestSetParser
from symbolic_data.paths import get_path, get_root, substitute_root_path
from symbolic_data.config_io import load_config, save_config
