"""symbolic_data -- the model-agnostic symbolic-regression data layer.

The public stack: one ``Problem`` (the central unit); a ``Catalog`` (level 1) that supplies
expressions plus their intrinsic sampling -- either a declarative ``ProblemCatalog`` (resolved by
reference via the versioned HF :mod:`~symbolic_data.resolver`) or an on-the-fly ``GenerativeCatalog``
(e.g. ``LampleChartonCatalog``, which grows random operator trees); and one ``ProblemSource``
(level 2) that samples a catalog into ``Problem``s under a usage policy (draw method,
support/validation counts, noise, problems-per-expression, holdouts/filters, materialization).
Plus the distribution vocabulary (incl. the ``fastsrb`` distribution) and the extensibility
``Registry``. Depends only on simplipy + numpy/sklearn.
"""
from symbolic_data.errors import NoValidSampleFoundError, CatalogEntryError
from symbolic_data.distributions import get_distribution, fastsrb_dist, DISTRIBUTIONS, BASE_DISTRIBUTIONS
from symbolic_data.prior_factory import build_prior_callable
from symbolic_data.registry import Registry
from symbolic_data.problem import Problem
from symbolic_data.catalog import Catalog, ProblemCatalog, CatalogEntry, RealizedExpression, load_catalog, CATALOGS
from symbolic_data.generative import GenerativeCatalog, LampleChartonCatalog, build_catalog, register_generative_catalog
from symbolic_data.source import ProblemSource
from symbolic_data.tensor_ops import mask_unused_variable_columns
from symbolic_data.paths import get_path, get_root, substitute_root_path
from symbolic_data.config_io import load_config, save_config
