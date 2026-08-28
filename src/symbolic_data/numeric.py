"""The one name for the width every realized array carries.

One name rather than a literal at each site, because the failure mode has no signature:
a narrowing is silent, and an array that has been through a narrower type and back is
still ``float64``-typed, still finite, still the right shape -- it has simply lost
mantissa bits, and nothing downstream can tell.

Realized support points, targets, noisy targets and sampled literals all carry this dtype.
Validity is judged in it too: a point whose target is not finite at this width is rejected,
so nothing ships as ``inf`` in a stored :class:`~symbolic_data.problem.Problem`.

Frozen ``.npz`` catalogs record the width they were written at in their ``_meta`` blob and
expose it as :attr:`~symbolic_data.catalog.ProblemCatalog.storage_dtype`; a catalog that
records nothing reads back as ``"float32"``, so a mixed corpus is detectable.
"""
import numpy as np

#: The dtype every realized array is stored and judged in.
STORAGE_DTYPE: np.dtype = np.dtype(np.float64)

#: The value written into a catalog's ``_meta`` blob, so the width a corpus was built at
#: is recoverable from the file itself.
STORAGE_DTYPE_MARKER: str = "float64"
