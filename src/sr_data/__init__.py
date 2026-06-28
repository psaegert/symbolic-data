"""sr-data -- the model-agnostic symbolic-regression data layer.

Skeleton/expression sampling, priors, (X, y) support sampling, holdout management, and
dataset construction -- carved out of flash-ansr so symbolic-regression methods and the
srbf eval framework share one data substrate. Depends only on simplipy + numpy/sklearn.
"""
from sr_data.skeleton_pool import SkeletonPool, NoValidSampleFoundError
from sr_data.holdout import HoldoutManager
from sr_data.skeleton_sampling import SkeletonSampler
from sr_data.support_sampling import SupportSampler, SupportSamplingError
from sr_data.distributions import get_distribution, DISTRIBUTIONS, BASE_DISTRIBUTIONS
from sr_data.prior_factory import build_prior_callable
from sr_data.registry import Registry
from sr_data.samples import Sample, sample_from_skeleton, iter_samples
from sr_data.tensor_ops import mask_unused_variable_columns
from sr_data.datasets import load_benchmark, BENCHMARKS
from sr_data.benchmarks import FastSRBBenchmark
from sr_data.paths import get_path, get_root, substitute_root_path
from sr_data.config_io import load_config, save_config
