"""The one name for the width every realized array carries.

One name rather than a literal at each site, because the failure mode has no signature:
a narrowing is silent, and an array that has been through ``float32`` and back is still
``float64``-typed, still finite, still the right shape -- it has simply lost 29 mantissa
bits and nothing downstream can tell.

Until the v25 migration this was ``float32``, because the consumer's boundary was: the
support tensors and the 32-bit ``<ieee754>`` constants format on the flash-ansr side.
That boundary is gone (flash-ansr 6ce8054 serializes constants as 8 IEEE-754 bytes at
binary64), so the generator no longer snaps to the f32 grid, no longer judges validity
there, and no longer rejects a finite value for overflowing a format nothing uses.

Frozen ``.npz`` catalogs are NOT rebuilt (owner ruling Q4): rebuilding would move every
stored value by up to one f32 ulp and invalidate every number ever published against
them. They carry a ``storage_dtype`` marker instead, so a mixed corpus is detectable.
"""
import numpy as np

#: The dtype every realized array is stored and judged in.
STORAGE_DTYPE: np.dtype = np.dtype(np.float64)

#: The value written into a catalog's ``_meta`` blob so a corpus built before the v25
#: widening is distinguishable from one built after it (owner ruling Q4).
STORAGE_DTYPE_MARKER: str = "float64"
