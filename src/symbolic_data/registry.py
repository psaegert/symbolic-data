"""A minimal, extensible ``name -> callable`` registry for symbolic_data.

Used for the parts of the data layer that select an implementation *by name* from a
config (currently: probability distributions). Third-party packages can add
implementations without editing symbolic_data, either

* in-process via the ``@REGISTRY.register("name")`` decorator, or
* across packages via ``importlib.metadata`` entry points in the group
  ``symbolic_data.<kind>`` (e.g. ``symbolic_data.distributions``).

Collision policy (intentionally permissive but loud):

* a direct ``register`` of an existing name *warns and keeps the existing one*
  unless ``overwrite=True`` is passed;
* an entry-point that shadows an already-registered name is *ignored with a
  warning* (first-registered wins), so a third-party typo cannot silently
  replace a builtin.

Reproducibility escape hatch: ``get(name, strict_builtins=True)`` resolves only
the seeded builtins and never triggers entry-point loading, so a run can pin
itself to the shipped implementations regardless of what is installed.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable


class Registry:
    """A named collection of callables, extensible via decorator or entry points."""

    def __init__(self, kind: str, *, entry_point_group: str | None = None) -> None:
        self._kind = kind
        self._entry_point_group = entry_point_group
        self._fns: dict[str, Callable[..., Any]] = {}
        self._builtins: set[str] = set()
        self._entry_points_loaded = False

    # -- registration ----------------------------------------------------------
    def register(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        *,
        overwrite: bool = False,
    ) -> Callable[..., Any]:
        """Register ``fn`` under ``name``. Usable as ``@reg.register("x")`` or ``reg.register("x", fn)``."""
        def _do(func: Callable[..., Any]) -> Callable[..., Any]:
            key = name.lower()
            if key in self._fns and not overwrite:
                warnings.warn(
                    f"{self._kind} {name!r} is already registered; "
                    "pass overwrite=True to shadow it.",
                    stacklevel=2,
                )
                return func
            self._fns[key] = func
            return func

        return _do if fn is None else _do(fn)

    def register_builtin(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Seed a builtin: always available, including under ``strict_builtins=True``."""
        key = name.lower()
        self._fns[key] = fn
        self._builtins.add(key)
        return fn

    # -- lookup ----------------------------------------------------------------
    def get(self, name: str, *, strict_builtins: bool = False) -> Callable[..., Any]:
        """Return the callable registered under ``name`` (case-insensitive)."""
        key = name.lower()
        if strict_builtins:
            if key in self._builtins:
                return self._fns[key]
            raise KeyError(self._unknown_message(name, strict_builtins=True))
        self._load_entry_points()
        try:
            return self._fns[key]
        except KeyError:
            raise KeyError(self._unknown_message(name)) from None

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        key = name.lower()
        if key in self._fns:
            return True
        self._load_entry_points()
        return key in self._fns

    def names(self, *, builtins_only: bool = False) -> list[str]:
        """Sorted registered names. ``builtins_only`` skips entry-point resolution."""
        if builtins_only:
            return sorted(self._builtins)
        self._load_entry_points()
        return sorted(self._fns)

    # -- entry points ----------------------------------------------------------
    def _load_entry_points(self) -> None:
        if self._entry_points_loaded or self._entry_point_group is None:
            return
        # Set the flag FIRST so a broken plugin cannot retrigger discovery on every call.
        self._entry_points_loaded = True
        from importlib.metadata import entry_points

        for ep in entry_points(group=self._entry_point_group):
            key = ep.name.lower()
            if key in self._fns:
                warnings.warn(
                    f"entry-point {self._kind} {ep.name!r} shadows an existing "
                    "registration; keeping the existing one (first wins).",
                    stacklevel=2,
                )
                continue
            try:
                self._fns[key] = ep.load()
            except Exception as exc:  # a broken third-party plugin must not break core
                warnings.warn(
                    f"failed to load {self._kind} entry-point {ep.name!r}: {exc}",
                    stacklevel=2,
                )

    def _unknown_message(self, name: str, *, strict_builtins: bool = False) -> str:
        if strict_builtins:
            return (
                f"Unknown {self._kind}: {name!r} (strict-builtins). "
                f"Builtins: {sorted(self._builtins)}"
            )
        return f"Unknown {self._kind}: {name!r}. Registered: {self.names()}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Registry(kind={self._kind!r}, builtins={sorted(self._builtins)})"
