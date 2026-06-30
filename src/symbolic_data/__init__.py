"""symbolic_data -- the model-agnostic symbolic-regression data layer.

The public stack: one ``Problem`` (the central unit), one declarative ``ProblemCatalog`` (level 1,
resolved by reference via the versioned HF :mod:`~symbolic_data.resolver`), and one
``ProblemSource`` (level 2) that turns a catalog / generator / inline problems into ``Problem``s
under a usage policy (draw method, support/validation counts, noise, problems-per-expression,
holdouts/filters, materialization). Plus the distribution vocabulary (incl. the ``fastsrb``
distribution) and the extensibility ``Registry``. Depends only on simplipy + numpy/sklearn.

The skeleton-sampling machinery is an internal implementation detail of generate-mode
``ProblemSource`` and is no longer part of the public API.
"""
from symbolic_data.errors import NoValidSampleFoundError
from symbolic_data.distributions import get_distribution, fastsrb_dist, DISTRIBUTIONS, BASE_DISTRIBUTIONS
from symbolic_data.prior_factory import build_prior_callable
from symbolic_data.registry import Registry
from symbolic_data.problem import Problem
from symbolic_data.catalog import ProblemCatalog, CatalogEntry, load_catalog, CATALOGS
from symbolic_data.source import ProblemSource
from symbolic_data.tensor_ops import mask_unused_variable_columns
from symbolic_data.paths import get_path, get_root, substitute_root_path
from symbolic_data.config_io import load_config, save_config
