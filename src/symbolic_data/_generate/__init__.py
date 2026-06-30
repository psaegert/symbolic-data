"""Private generate engine for on-the-fly ProblemSource sampling.

NOT a public API. These modules (the skeleton sampler, support sampler, holdout grid, and the
procedural catalog) are the internal machinery that generate-mode
:class:`~symbolic_data.source.ProblemSource` drives. The public data-layer surface is
``Problem`` / ``ProblemCatalog`` / ``ProblemSource``; nothing here is exported from
``symbolic_data``.
"""
