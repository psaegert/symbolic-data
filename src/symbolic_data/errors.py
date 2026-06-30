"""Shared exceptions for the data layer (no internal imports -> importable everywhere)."""


class NoValidSampleFoundError(Exception):
    """A catalog could not realize a valid sample for an expression within its retry budget.

    Raised by a :class:`~symbolic_data.catalog.Catalog`'s ``realize`` (and by the generative
    skeleton sampler) on a *transient* failure; the caller retries up to ``max_trials`` and then
    emits a placeholder :class:`~symbolic_data.problem.Problem`.
    """


class CatalogEntryError(Exception):
    """A catalog entry is *permanently* unrealizable (e.g. it fails to compile).

    Distinct from :class:`NoValidSampleFoundError`: there is no point retrying, so the caller
    emits a placeholder immediately rather than exhausting its trial budget.
    """
