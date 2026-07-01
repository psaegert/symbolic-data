"""Generative (on-the-fly) catalogs: the public counterpart to a declarative ``ProblemCatalog``.

A :class:`GenerativeCatalog` produces *fresh* expressions on demand rather than holding a fixed
set; a :class:`~symbolic_data.source.ProblemSource` samples from it exactly as it samples from a
declarative catalog. :class:`LampleChartonCatalog` is the concrete generator: it grows random
unary-binary operator trees (the Lample-Charton recipe) and realizes their support data.

The skeleton/support/holdout machinery is private (``_generate``); this module is the public face.
"""
import os
import re
import warnings
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

from types import CodeType
from typing import Any, Callable, Iterator, Mapping, Sequence

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from simplipy import SimpliPyEngine, normalize_expression, normalize_skeleton
from simplipy.utils import explicit_constant_placeholders, numbers_to_constant, substitude_constants

from symbolic_data.config_io import load_config, save_config
from symbolic_data.paths import substitute_root_path
from symbolic_data.sympy_timeout import _sympy_simplify_with_timeout
from simplipy.utils import codify
from symbolic_data.prior_factory import build_prior_callable
from symbolic_data._generate.holdout import HoldoutManager
from symbolic_data._generate.skeleton_sampling import SkeletonSampler
from symbolic_data._generate.support_sampling import SupportSampler, SupportSamplingError
from symbolic_data.token_ops import flatten_nested_list
from symbolic_data.errors import NoValidSampleFoundError
from symbolic_data.catalog import Catalog, ProblemCatalog, RealizedExpression


def _gt_metadata(skeleton: Sequence[str], literals: Any) -> tuple[list[str] | None, int | None]:
    """Normalized ground-truth expression (constants substituted) + its token-length complexity."""
    skeleton_list = normalize_skeleton(skeleton)
    if skeleton_list is None:
        return None, None
    expression_tokens = substitude_constants(list(skeleton_list), values=literals, inplace=False)
    expression = normalize_expression(expression_tokens)
    complexity = len(expression_tokens) if expression_tokens else None
    return expression, complexity


@dataclass
class GeneratedEntry:
    """One freshly generated skeleton (the unit a GenerativeCatalog yields to ProblemSource)."""

    skeleton: tuple[str, ...]
    code: CodeType
    constants: list[str]              # constant-placeholder token names
    variables: list[str]             # the catalog's variable column order


class GenerativeCatalog(Catalog, ABC):
    """Abstract base for on-the-fly catalogs (generate expressions instead of holding a fixed set).

    Concretes implement :meth:`sample_skeleton` (the generation algorithm) and the
    :class:`~symbolic_data.catalog.Catalog` interface (``iter_entries`` / ``realize``). Unlike a
    declarative catalog, a generative one is *infinite*: ``iter_entries(size=None)`` streams forever.
    """

    @abstractmethod
    def sample_skeleton(self, new: bool = False, decontaminate: bool = True, rng: np.random.Generator | None = None) -> tuple[tuple[str, ...], CodeType, list[str]]:
        """Sample one raw skeleton (operator template), its compiled code, and its constant tokens."""

    def is_finite(self) -> bool:
        return False


def _constantify_skeleton(skeleton: list[str]) -> list[str]:
    """Replace mult<N>/div<N> tokens with multiplication by a constant."""
    result: list[str] = []
    for token in skeleton:
        m = re.match(r'^(mult|div)(\d+)$', token)
        if m:
            result.extend(['*', m.group(2)])
        else:
            result.append(token)
    return result


class LampleChartonCatalog(GenerativeCatalog):
    '''
    A generative catalog that grows random unary-binary operator trees (Lample-Charton recipe).

    Samples expression 'skeletons' (prefix expressions with placeholders for constants) on the fly,
    optionally caching a finite set of them, and realizes each into support data. It is the
    on-the-fly counterpart to a declarative :class:`~symbolic_data.catalog.ProblemCatalog`.

    Parameters
    ----------
    simplipy_engine : SimpliPyEngine
        The expression space to operate in.
    sample_strategy : dict[str, Any]
        The strategy to use for sampling skeletons.
    literal_prior : dict[str, Any] | list[dict[str, Any]]
        The prior distribution for the literals.
    support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
        Legacy override for the support prior. When provided, it will overwrite the
        corresponding entry in ``support_sampler_config``.
    support_scale_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
        Legacy override for the support-scale prior. Merged into ``support_sampler_config``
        when supplied.
    n_support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
        Legacy override for the support-count prior. Merged into ``support_sampler_config``
        when supplied.
    support_sampler_config : dict[str, Any] or None, optional
        Unified configuration describing how support points are generated (priors,
        uniqueness constraints, quantized behaviour, etc.).
    holdout_pools : Sequence[LampleChartonCatalog | str] or None, optional
        LampleChartonCatalogs or paths to pools to exclude when sampling.
    allow_nan : bool, optional
        Whether to allow NaNs in the support points.
    simplify : bool or str, optional
        Whether and how to simplify sampled skeletons.
        ``True`` uses SimpliPy, ``'sympy'`` uses SymPy (with timeout), ``False`` disables simplification.
    '''

    def __init__(
            self,
            simplipy_engine: SimpliPyEngine,
            sample_strategy: dict[str, Any],
            literal_prior: dict[str, Any] | list[dict[str, Any]] | Callable,
            variables: list[str],
            support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            support_scale_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            n_support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            support_sampler_config: dict[str, Any] | None = None,
            operator_weights: dict[str, float] | None = None,
            holdout_pools: Sequence["LampleChartonCatalog | str"] | None = None,
            allow_nan: bool = False,
            simplify: bool | str = True,
            name: str = "lample_charton",
            decontaminate: bool = True) -> None:
        self.name = name
        self.decontaminate = decontaminate
        self.simplipy_engine = simplipy_engine
        self.sample_strategy = sample_strategy
        self.variables = variables
        self.n_variables = len(self.variables)
        self.operator_weights = operator_weights or {op: 1.0 for op in self.simplipy_engine.operator_arity.keys()}

        self.holdout_manager = HoldoutManager(n_variables=self.n_variables, allow_nan=allow_nan)

        self.holdout_pools: list["LampleChartonCatalog | str"] = []
        for holdout_pool in holdout_pools or []:
            self.register_holdout_pool(holdout_pool)

        self.skeletons: set[tuple[str]] = set()
        self.skeleton_codes: dict[tuple[str], tuple[CodeType, list[str]]] = {}

        self.skeleton_sampler = SkeletonSampler(
            simplipy_engine=self.simplipy_engine,
            sample_strategy=self.sample_strategy,
            variables=self.variables,
            operator_weights=self.operator_weights,
        )

        if isinstance(literal_prior, (dict, list)):
            self.literal_prior_config = literal_prior
            self.literal_prior: Callable = build_prior_callable(literal_prior)
        elif callable(literal_prior):
            self.literal_prior = literal_prior
        else:
            raise ValueError("literal_prior must be either a dict, list of dicts, or a callable")

        support_config = deepcopy(support_sampler_config) if support_sampler_config is not None else {}

        if support_prior is not None:
            support_config["support_prior"] = support_prior
        if support_scale_prior is not None:
            support_config["support_scale_prior"] = support_scale_prior
        if n_support_prior is not None:
            support_config["n_support_prior"] = n_support_prior

        self.support_sampler_config = support_config

        self.allow_nan = allow_nan
        self.simplify = simplify

        independent_dims = self.sample_strategy.get('independent_dimensions', False)
        self.support_sampler = SupportSampler(
            n_variables=self.n_variables,
            independent_dimensions=independent_dims,
            config=self.support_sampler_config,
        )

        self.operator_probs: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "LampleChartonCatalog":
        '''
        Create a LampleChartonCatalog from a configuration dictionary or file.

        Parameters
        ----------
        config : dict or str
            The configuration dictionary or path to the configuration file.

        Returns
        -------
        LampleChartonCatalog
            The LampleChartonCatalog object.
        '''
        config_ = load_config(config)

        # If the config is a string, convert relative paths within the config to absolute paths
        if isinstance(config, str) and isinstance(config_["simplipy_engine"], str):
            if config_["simplipy_engine"].startswith('.'):
                config_["simplipy_engine"] = os.path.join(os.path.dirname(config), config_["simplipy_engine"])

        support_sampler_cfg = deepcopy(config_.get("support_sampler")) if config_.get("support_sampler") else {}
        for key in ("support_prior", "support_scale_prior", "n_support_prior"):
            if key in config_ and key not in support_sampler_cfg:
                support_sampler_cfg[key] = config_[key]

        catalog = cls(
            simplipy_engine=SimpliPyEngine.load(config_["simplipy_engine"], install=True),
            sample_strategy=config_["sample_strategy"],
            literal_prior=config_["literal_prior"],
            variables=config_["variables"],
            support_sampler_config=support_sampler_cfg,
            operator_weights=config_.get("operator_weights"),
            holdout_pools=config_.get("holdout_pools", []),
            allow_nan=config_.get("allow_nan", False),
            simplify=config_.get("simplify", True),
            name=config_.get("name", "lample_charton"),
            decontaminate=config_.get("decontaminate", True),
        )
        # Optional inline FROZEN skeletons (a fixed catalog distributed as one self-contained spec --
        # e.g. a held-out validation set): the recipe still drives X/y sampling, but the skeleton set
        # is fixed (sample_skeleton(new=False) draws from these), not generated on the fly.
        frozen = config_.get("skeletons")
        if frozen:
            catalog.skeletons = {tuple(s) for s in frozen}
            catalog.skeleton_codes = catalog.compile_codes()
        return catalog

    @classmethod
    def from_dict(
            cls,
            skeletons: set[tuple[str]],
            simplipy_engine: SimpliPyEngine,
            sample_strategy: dict[str, Any],
            literal_prior: dict[str, Any] | list[dict[str, Any]] | Callable,
            variables: list[str],
            support_sampler_config: dict[str, Any] | None = None,
            support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            support_scale_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            n_support_prior: dict[str, Any] | list[dict[str, Any]] | Callable | None = None,
            operator_weights: dict[str, float] | None = None,
            skeleton_codes: dict[tuple[str], tuple[CodeType, list[str]]] | None = None,
            holdout_pools: Sequence["LampleChartonCatalog | str"] | None = None,
            allow_nan: bool = False,
            simplify: bool = True) -> "LampleChartonCatalog":
        '''
        Create a LampleChartonCatalog from a set of skeletons.

        Parameters
        ----------
        skeletons : set[tuple[str]]
            The set of skeletons to include in the pool.
        simplipy_engine : SimpliPyEngine
            The expression space to operate in.
        sample_strategy : dict[str, Any]
            The strategy to use for sampling skeletons.
        literal_prior : dict[str, Any] | list[dict[str, Any]]
            The prior distribution for the literals.
        variables : list[str]
            The variables to use in the expressions.
        support_sampler_config : dict[str, Any] or None, optional
            Unified support sampling configuration. If provided alongside the legacy
            priors below, the legacy values overwrite matching keys within this config.
        support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
            Optional override for the support prior.
        support_scale_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
            Optional override for the support-scale prior.
        n_support_prior : dict[str, Any] | list[dict[str, Any]] | Callable or None, optional
            Optional override for the support-count prior.
        operator_weights : dict[str, float] or None, optional
            A dictionary mapping operators to their weights.
        skeleton_codes : dict[tuple[str], tuple[CodeType, list[str]]] or None, optional
            A dictionary mapping skeletons to their compiled codes.
        holdout_pools : Sequence[LampleChartonCatalog | str] or None, optional
            LampleChartonCatalogs or paths to pools to exclude when sampling.
        allow_nan : bool, optional
            Whether to allow NaNs in the support points.
        simplify : bool, optional
            Whether to simplify sampled skeletons.

        Returns
        -------
        LampleChartonCatalog
            The LampleChartonCatalog object.
        '''
        catalog = cls(
            simplipy_engine=simplipy_engine,
            sample_strategy=sample_strategy,
            literal_prior=literal_prior,
            variables=variables,
            support_prior=support_prior,
            support_scale_prior=support_scale_prior,
            n_support_prior=n_support_prior,
            support_sampler_config=support_sampler_config,
            operator_weights=operator_weights,
            holdout_pools=holdout_pools,
            allow_nan=allow_nan,
            simplify=simplify
        )

        catalog.skeletons = skeletons

        if skeleton_codes is not None:
            catalog.skeleton_codes = skeleton_codes
        else:
            catalog.skeleton_codes = catalog.compile_codes()

        return catalog

    def compile_codes(self, verbose: bool = False) -> dict[tuple[str], tuple[CodeType, list[str]]]:
        '''
        Compile the skeletons in the pool into executable code.

        Parameters
        ----------
        verbose : bool, optional
            Whether to display a progress bar.

        Returns
        -------
        dict[tuple[str], tuple[CodeType, list[str]]]
            A dictionary mapping skeletons to their compiled codes.
        '''
        codes = {}
        for skeleton in tqdm(self.skeletons, desc="Compiling Skeletons", disable=not verbose, smoothing=0.0):
            # Codify the Expression
            executable_prefix_expression = self.simplipy_engine.operators_to_realizations(skeleton)
            prefix_expression_with_constants, constants = explicit_constant_placeholders(executable_prefix_expression, inplace=True)
            code_string = self.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.variables + constants)

            codes[skeleton] = (code, constants)

        return codes

    def __contains__(self, skeleton: tuple[str] | list[str]) -> bool:
        '''
        Check if a skeleton is in the pool.

        Parameters
        ----------
        skeleton : tuple[str] or list[str]
            The skeleton to check.

        Returns
        -------
        bool
            Whether the skeleton is in the pool.
        '''
        return tuple(skeleton) in self.skeletons

    def is_held_out(self, skeleton: tuple[str] | list[str], constants: list[str], code: CodeType | None = None) -> bool:
        '''
        Check if a skeleton is held out from the pool.

        Parameters
        ----------
        skeleton : tuple[str] or list[str]
            The skeleton to check.
        constants : list[str]
            The constants used in the skeleton.
        code : CodeType or None, optional
            The compiled code for the skeleton. If not provided, it will be compiled.

        Returns
        -------
        bool
            Whether the skeleton is held out.
        '''
        if constants is None:
            raise ValueError("Need constants for test of functional equivalence")

        no_constant_expression = self.get_structural_prototype(skeleton)

        if code is None:
            executable_prefix_expression = self.simplipy_engine.operators_to_realizations(no_constant_expression)
            prefix_expression_with_constants, constants = explicit_constant_placeholders(executable_prefix_expression, inplace=True)
            code_string = self.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, self.variables + constants)

        compiled_fn = self.simplipy_engine.code_to_lambda(code)

        try:
            return self.holdout_manager.is_held_out(
                no_constant_expression,
                compiled_fn,
                num_constants=len(constants),
            )
        except (OverflowError, NameError):
            warnings.warn(
                f"Overflow/Name error during holdout evaluation; assuming held out "
                f"(skeleton={skeleton}, constants={constants})",
                RuntimeWarning,
                stacklevel=2,
            )
            return True

    @property
    def holdout_skeletons(self) -> set[tuple[str, ...]]:
        return self.holdout_manager.skeleton_hashes

    @property
    def holdout_y(self) -> set[tuple[float, ...] | tuple[tuple[float, ...], ...]]:
        return self.holdout_manager.expression_images

    @property
    def holdout_X(self) -> np.ndarray:
        return self.holdout_manager.holdout_X

    @property
    def holdout_C(self) -> np.ndarray:
        return self.holdout_manager.holdout_C

    def get_structural_prototype(self, expression: list[str] | tuple[str, ...], verbose: bool = False, debug: bool = False) -> list[str]:
        stack: list = []
        i = len(expression) - 1

        if debug:
            print(f'Input expression: {expression}')

        while i >= 0:
            token = expression[i]

            if debug:
                print(f'Stack: {stack}')
                print(f'Processing token {token}')

            if token in self.simplipy_engine.operator_arity_compat or token in self.simplipy_engine.operator_aliases:
                operator = self.simplipy_engine.operator_aliases.get(token, token)
                arity = self.simplipy_engine.operator_arity_compat[operator]
                operands = list(reversed(stack[-arity:]))

                if any(operand[0] == '<constant>' for operand in operands):
                    if verbose:
                        print('Removing constant')

                    non_num_operands = [operand for operand in operands if operand[0] != '<constant>']

                    if len(non_num_operands) == 0:
                        new_term = '<constant>'
                    elif len(non_num_operands) == 1:
                        new_term = non_num_operands[0]
                    else:
                        raise NotImplementedError('Removing a constant from n-operand operator is not implemented')

                    _ = [stack.pop() for _ in range(arity)]
                    stack.append([new_term])
                    i -= 1
                    continue

                _ = [stack.pop() for _ in range(arity)]
                stack.append([operator, operands])

            else:
                stack.append([token])

            i -= 1

        return flatten_nested_list(stack, reverse=True)

    def register_holdout_pool(self, holdout_pool: "LampleChartonCatalog | str") -> None:
        '''
        Register a holdout pool to exclude from sampling: Cache the skeletons and their images to compare against when sampling.

        Parameters
        ----------
        holdout_pool : LampleChartonCatalog or str
            The holdout pool object or path to register.
        '''
        if isinstance(holdout_pool, str):
            if holdout_pool in self.holdout_pools:
                return
            resolved = substitute_root_path(holdout_pool)
            # a saved-catalog DIRECTORY (legacy) -> load it; otherwise a ref/name/inline -> build_catalog
            # (resolves a name[@version] via HF, a config path, or an inline spec).
            holdout_pool_obj = LampleChartonCatalog.load(resolved)[1] if os.path.isdir(resolved) else build_catalog(resolved)
            self.holdout_pools.append(holdout_pool)
        else:
            if any(existing is holdout_pool for existing in self.holdout_pools if not isinstance(existing, str)):
                return

            holdout_pool_obj = holdout_pool
            self.holdout_pools.append(holdout_pool_obj)

        # A generative (skeleton-bearing) holdout exposes its own skeletons + engine/variables; a
        # declarative ProblemCatalog (e.g. the canonical `fastsrb`) does not, so derive its structural
        # prototypes from its expressions in THIS catalog's space (self.simplipy_engine / self.variables).
        if isinstance(holdout_pool_obj, GenerativeCatalog):
            items = [
                (holdout_pool_obj.get_structural_prototype(sk), holdout_pool_obj.simplipy_engine,
                 holdout_pool_obj.variables, holdout_pool_obj.n_variables)
                for sk in holdout_pool_obj.skeletons
            ]
        else:
            items = []
            for entry in holdout_pool_obj.iter_entries(np.random.default_rng()):
                expression = getattr(entry, "prepared", None) or getattr(entry, "raw", None)
                if expression is None:
                    continue
                prefix = self.simplipy_engine.infix_to_prefix(expression)
                # normalize_skeleton canonicalizes the declarative source's variable names (e.g. v1->x1)
                # into THIS catalog's space and abstracts numeric literals to <constant>; then take the
                # structural prototype (constants removed) -- the same form the generative path registers.
                canonical = normalize_skeleton(prefix)
                if canonical is None:
                    continue
                items.append((self.get_structural_prototype(canonical), self.simplipy_engine,
                              self.variables, self.n_variables))

        for no_constant_expression, engine, variables, n_variables in items:
            executable_prefix_expression = engine.operators_to_realizations(no_constant_expression)
            prefix_expression_with_constants, constants = explicit_constant_placeholders(executable_prefix_expression, inplace=True)
            code_string = engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
            code = codify(code_string, variables + constants)
            compiled_fn = engine.code_to_lambda(code)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                self.holdout_manager.register_skeleton(
                    no_constant_expression,
                    compiled_fn,
                    num_constants=len(constants),
                    n_variables=n_variables,
                )

    def clear_holdouts(self) -> None:
        """Remove all registered holdout pools and associated constraints."""
        self.holdout_pools = []
        self.holdout_manager = HoldoutManager(n_variables=self.n_variables, allow_nan=self.allow_nan)

    def save(self, directory: str, config: dict[str, Any] | str | None = None, reference: str = 'relative', recursive: bool = True) -> None:
        '''
        Save the catalog to a directory.

        Parameters
        ----------
        directory : str
            The directory to save the catalog to.
        config : dict or str or None, optional
            The configuration dictionary or path to the configuration file. If None, the model will be saved without a config file.
        reference : str, optional
            The reference path to use for saving the model.
        recursive : bool, optional
            Whether to save the model recursively (i.e. save the config and holdout pools).
        '''
        directory = substitute_root_path(directory)

        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'skeletons.pkl'), 'wb') as pool_file:
            pickle.dump(self.skeletons, pool_file)

        # Copy the config to the directory for best portability
        if config is None:
            warnings.warn("No config specified, saving the model without a config file. Loading the model will require manual configuration.")
        else:
            save_config(load_config(config, resolve_paths=True), directory=directory, filename='catalog.yaml', reference=reference, recursive=recursive, resolve_paths=True)

    @classmethod
    def load(cls, directory: str, verbose: bool = True) -> tuple[dict[str, Any], "LampleChartonCatalog"]:
        '''
        Load a catalog from a directory.

        Parameters
        ----------
        directory : str
            The directory to load the catalog from.
        verbose : bool, optional
            Whether to display a progress bar.

        Returns
        -------
        dict[str, Any], LampleChartonCatalog
            The configuration dictionary and the LampleChartonCatalog object.
        '''
        config_path = os.path.join(directory, 'catalog.yaml')
        resolved_directory = substitute_root_path(directory)

        pool = cls.from_config(config_path)

        with open(os.path.join(resolved_directory, 'skeletons.pkl'), 'rb') as pool_file:
            pool.skeletons = pickle.load(pool_file)

            pool.skeleton_codes = pool.compile_codes(verbose=verbose)

        return load_config(config_path), pool

    def _sympy_simplify_skeleton(self, skeleton: list[str], rng: np.random.Generator) -> list[str]:
        """Simplify a skeleton using SymPy with a subprocess timeout."""
        # Replace mult/div tokens with arithmetic equivalents
        expression = _constantify_skeleton(list(skeleton))

        # Extract constant placeholders
        expression, constants = explicit_constant_placeholders(expression)

        # Convert prefix to infix for SymPy
        infix = self.simplipy_engine.prefix_to_infix(expression, power='**')

        # Replace constant placeholders with random numerical values
        for c in constants:
            infix = infix.replace(c, str(rng.uniform(-10, 10)))

        # Run SymPy simplification with timeout
        result = _sympy_simplify_with_timeout(infix, timeout_seconds=1.0)
        if result is None:
            raise NoValidSampleFoundError("SymPy simplification timed out or failed")

        simplified_infix, _ = result

        # Translate SymPy function names back
        simplified_infix = simplified_infix.replace('Abs', 'abs')

        # Parse back to prefix notation
        prefix = self.simplipy_engine.parse(simplified_infix)

        # Convert numeric literals back to constant placeholders
        prefix = numbers_to_constant(prefix, inplace=True)

        if any(forbidden_token in prefix for forbidden_token in ['float("inf")', 'float("-inf")', 'float("nan")', 'zoo', 'nan', 'oo']):
            raise NoValidSampleFoundError(f"SymPy result contains forbidden tokens: {prefix}")

        return prefix

    def sample_skeleton(self, new: bool = False, decontaminate: bool = True, rng: np.random.Generator | None = None) -> tuple[tuple[str], CodeType, list[str]]:
        '''
        Sample a skeleton from the pool.

        Parameters
        ----------
        new : bool, optional
            Whether to sample a new skeleton or reuse an existing one.

        Returns
        -------
        tuple[str], CodeType, list[str]
            The skeleton, its compiled code, and the constants used in the skeleton.
        '''
        rng = rng if rng is not None else np.random.default_rng()
        if len(self.skeletons) == 0 or new:
            for _ in range(self.sample_strategy['max_tries']):
                match self.sample_strategy['n_operator_distribution']:
                    case "equiprobable_lengths":
                        n_operators = rng.integers(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)
                    case "length_proportional":
                        if self.operator_probs is None:
                            self.operator_probs = np.arange(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)**self.sample_strategy['power']
                            self.operator_probs = self.operator_probs / self.operator_probs.sum()
                        n_operators = rng.choice(
                            range(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1),
                            p=self.operator_probs)
                    case "length_exponential":
                        if self.operator_probs is None:
                            self.operator_probs = np.exp(np.arange(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1)**self.sample_strategy['power'] / self.sample_strategy['lambda'])
                            self.operator_probs = self.operator_probs / self.operator_probs.sum()
                        n_operators = rng.choice(
                            range(self.sample_strategy['min_operators'], self.sample_strategy['max_operators'] + 1),
                            p=self.operator_probs)
                    case _:
                        raise ValueError(f"Invalid n_operator_distribution: {self.sample_strategy['n_operator_distribution']}")

                skeleton = self.skeleton_sampler.sample(n_operators, rng)
                if self.simplify is True:
                    try:
                        skeleton = self.simplipy_engine.simplify(skeleton, inplace=True, max_pattern_length=4)
                    except Exception as e:
                        raise NoValidSampleFoundError(f"Failed to simplify skeleton: {skeleton}") from e

                    if any(forbidden_token in skeleton for forbidden_token in ['float("inf")', 'float("-inf")', 'float("nan")']):
                        raise NoValidSampleFoundError(f"Skeleton contains forbidden tokens: {skeleton}")
                elif self.simplify == 'sympy':
                    try:
                        skeleton = self._sympy_simplify_skeleton(skeleton, rng)
                    except NoValidSampleFoundError:
                        raise
                    except Exception as e:
                        raise NoValidSampleFoundError(f"SymPy failed on skeleton: {skeleton}") from e

                if tuple(skeleton) not in self.skeletons and len(skeleton) <= self.sample_strategy['max_length']:
                    executable_prefix_expression = self.simplipy_engine.operators_to_realizations(skeleton)
                    prefix_expression_with_constants, constants = explicit_constant_placeholders(executable_prefix_expression, inplace=True)
                    try:
                        code_string = self.simplipy_engine.prefix_to_infix(prefix_expression_with_constants, realization=True)
                    except ValueError:
                        raise NoValidSampleFoundError(f"Malformed prefix expression after simplification: {skeleton}")
                    code = codify(code_string, self.variables + constants)

                    if not decontaminate or not self.is_held_out(skeleton, constants):
                        return tuple(skeleton), code, constants   # type: ignore
        else:
            skeletons_tuple = tuple(self.skeletons)
            skeleton = skeletons_tuple[int(rng.integers(len(skeletons_tuple)))]  # type: ignore
            code, constants = self.skeleton_codes[skeleton]  # type: ignore

            return skeleton, code, constants  # type: ignore

        raise NoValidSampleFoundError(f"Failed to sample a non-contaminated skeleton after {self.sample_strategy['max_tries']} retries")

    def sample_data(
            self, code: CodeType, n_constants: int = 0, n_support: int | None = None,
            support_prior: Callable | None = None, support_scale_prior: Callable | None = None,
            rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Sample support points and literals for an expression.

        Parameters
        ----------
        code : CodeType
            The compiled code for the expression.
        n_constants : int, optional
            The number of constants to sample.
        n_support : int or None, optional
            The number of support points to sample. If None, the number of support points will be sampled from the prior distribution.
        support_prior : Callable or None, optional
            The prior distribution for the support points. If None, the default support prior will be used.
        support_scale_prior : Callable or None, optional
            The prior distribution for the support scale. If None, the default support scale prior will be used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The support points, their images, and the literals.
        '''
        rng = rng if rng is not None else np.random.default_rng()
        expression_callable = self.simplipy_engine.code_to_lambda(code)
        if n_support is None:
            n_support = self.support_sampler.sample_n_support(rng=rng)

        for _ in range(self.sample_strategy['max_tries']):
            literals = self.literal_prior(size=n_constants, rng=rng).astype(np.float32)

            override_support_prior = SupportSampler.ensure_prior_callable(support_prior) if support_prior is not None else None
            override_support_scale = SupportSampler.ensure_prior_callable(support_scale_prior) if support_scale_prior is not None else None

            try:
                x_support = self.support_sampler.sample(
                    n_support=n_support,
                    support_prior=override_support_prior,
                    support_scale_prior=override_support_scale,
                    rng=rng,
                )
            except SupportSamplingError:
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                y_support = expression_callable(*x_support.T, *literals)

            if not isinstance(y_support, np.ndarray):
                y_support = np.full((n_support, 1), y_support, dtype=np.float32)

            if len(y_support) == 1:
                # Repeat y to match the shape of x
                y_support = np.repeat(y_support, x_support.shape[0])

            # Complex numbers are not supported
            if np.iscomplex(y_support).any() or y_support.dtype != np.float32:
                continue

            if not self.allow_nan:
                # If any of the support points are NaN, skip the expression
                if np.isnan(x_support).any() or np.isinf(x_support).any():
                    continue
                if np.isnan(y_support).any() or np.isinf(y_support).any():
                    continue
            elif np.isnan(y_support).all():
                # Even if NaNs are allowed, if all support points are NaN, skip the expression
                continue

            # All checks passed, break the loop
            break
        else:
            raise NoValidSampleFoundError(f"Failed to generate a valid expression after {self.sample_strategy['max_tries']} retries")

        return x_support, y_support.reshape(-1, 1), literals

    def sample(self, n_support: int | None = None, rng: np.random.Generator | None = None) -> dict:
        '''
        Sample a skeleton, support points, and literals.

        Parameters
        ----------
        n_support : int or None, optional
            The number of support points to sample. If None, the number of support points will be sampled from the prior distribution.

        Returns
        -------
        dict
            A dictionary containing the skeleton hash, code, constants, support points, and literals.
        '''
        rng = rng if rng is not None else np.random.default_rng()
        skeleton_hash, skeleton_code, skeleton_constants = self.sample_skeleton(rng=rng)
        x_support, y_support, literals = self.sample_data(skeleton_code, len(skeleton_constants), n_support, rng=rng)

        return {
            'skeleton_hash': skeleton_hash,
            'skeleton_code': skeleton_code,
            'skeleton_constants': skeleton_constants,
            'x_support': x_support,
            'y_support': y_support,
            'literals': literals
        }

    def create(self, size: int, verbose: bool = False, rng: np.random.Generator | None = None) -> None:
        '''
        Create a pool of skeletons of a given size.

        Parameters
        ----------
        size : int
            The number of skeletons to create.
        verbose : bool, optional
            Whether to display a progress bar.
        '''
        rng = rng if rng is not None else np.random.default_rng()
        n_skipped = 0
        n_created = len(self.skeletons)

        pbar = tqdm(total=size, desc="Generating skeletons", disable=not verbose, smoothing=0.0)

        while n_created < size:
            try:
                skeleton, code, constants = self.sample_skeleton(new=True, rng=rng)
            except NoValidSampleFoundError:
                n_skipped += 1
                pbar.set_postfix_str(f"Skipped: {n_skipped:,}")
                continue

            if not self.simplipy_engine.is_valid(skeleton):
                raise ValueError(f"Invalid skeleton: {skeleton}")

            if not isinstance(skeleton, tuple):
                skeleton = tuple(skeleton)
            self.skeletons.add(skeleton)
            self.skeleton_codes[skeleton] = (code, constants)
            n_created += 1

            pbar.update(1)
            pbar.set_postfix_str(f"Skipped: {n_skipped:,}")

            if n_created >= size:
                break

    # --- Catalog interface (sampled by ProblemSource) -----------------------------------------
    def iter_entries(self, rng: np.random.Generator, *, method: str = "iterate", size: int | None = None) -> Iterator[GeneratedEntry]:
        """Yield generated skeletons.

        ``method="iterate"`` (the DEFAULT, matching :meth:`Catalog.iter_entries`; BOUNDED):
        ``size`` is ``None`` -> iterate the existing fixed skeleton set ONCE (each skeleton once, e.g. a
        frozen validation pool), so ``list(...)`` terminates at ``len(skeletons)``; ``size=N`` ->
        generate a finite pool of ``N`` distinct skeletons and iterate it. On an OPEN catalog with
        NEITHER a fixed skeleton set NOR a ``size``, "iterate" is undefined (there is nothing bounded to
        yield) -> a clear ``ValueError`` instead of a silent unbounded stream a caller might ``list()``.

        ``method="procedural"`` (UNBOUNDED training stream; must be requested explicitly):
        ``size`` is ``None`` -> a fresh skeleton each draw (an EMPTY catalog) or the existing set sampled
        WITH REPLACEMENT (a pre-loaded one); ``size=N`` -> a finite pool. This is what
        :class:`ProblemSource` passes in generate mode; a direct ``list()`` of it never terminates.
        """
        if method == "procedural" and size is None:
            # Explicit unbounded stream (generate-mode streaming). Never the default, so a naive
            # `list(catalog.iter_entries(rng))` no longer hangs -- it takes the bounded path below.
            while True:
                try:
                    skeleton, code, constants = self.sample_skeleton(new=False, decontaminate=self.decontaminate, rng=rng)
                except NoValidSampleFoundError:
                    continue
                yield GeneratedEntry(skeleton=tuple(skeleton), code=code, constants=list(constants), variables=list(self.variables))
            return
        # Bounded paths: an explicit `size` generates a finite pool; otherwise iterate the fixed set.
        if size is not None:
            self.create(int(size), rng=rng)
        elif not self.skeletons:
            raise ValueError(
                "iter_entries(method='iterate') on an open generative catalog with no fixed skeleton "
                "set is unbounded and cannot be materialized: pass size=N to generate a finite pool, "
                "or method='procedural' for an explicit unbounded training stream."
            )
        if not self.skeleton_codes:
            self.skeleton_codes = self.compile_codes()
        for skeleton in sorted(self.skeletons):
            code, constants = self.skeleton_codes[skeleton]
            yield GeneratedEntry(skeleton=tuple(skeleton), code=code, constants=list(constants), variables=list(self.variables))

    def realize(self, entry: GeneratedEntry, n_points: int, rng: np.random.Generator, *, engine: Any = None, layout: str = "random") -> RealizedExpression:
        """Realize one generated skeleton into support data (its intrinsic support + literal sampling).

        ``engine``/``layout`` are ignored: a generative catalog carries its own engine and support
        sampler. Raises :class:`NoValidSampleFoundError` (retryable) when no valid support is found.
        """
        x_all, y_all, literals = self.sample_data(entry.code, len(entry.constants), n_support=n_points, rng=rng)
        if x_all.size == 0 or y_all.size == 0:
            raise NoValidSampleFoundError("empty support sample")
        expression, complexity = _gt_metadata(entry.skeleton, literals)
        return RealizedExpression(
            x=x_all, y=y_all,
            skeleton=tuple(entry.skeleton), expression=expression,
            constants=list(np.asarray(literals, dtype=np.float64).ravel().tolist()),
            variables=list(entry.variables), complexity=complexity, eq_id=None,
        )

    def split(self, train_size: float, random_state: int | None = None) -> tuple["LampleChartonCatalog", "LampleChartonCatalog"]:
        """
        Split the catalog into two disjoint pools randomly.

        Parameters
        ----------
        train_size : float
            The size of the first pool as a fraction of the original pool size.
        random_state : int or None, optional
            The random state to use for splitting.

        Returns
        -------
        tuple[LampleChartonCatalog, LampleChartonCatalog]
            The two disjoint catalogs.
        """
        train_keys, test_keys = train_test_split(list(self.skeletons), train_size=train_size, random_state=random_state)

        train_pool = LampleChartonCatalog.from_dict(
            set(train_keys),
            simplipy_engine=self.simplipy_engine,
            sample_strategy=self.sample_strategy,
            literal_prior=self.literal_prior,
            skeleton_codes={k: v for k, v in self.skeleton_codes.items() if k in train_keys},
            variables=self.variables,
            support_sampler_config=deepcopy(self.support_sampler_config),
            operator_weights=self.operator_weights,
            holdout_pools=self.holdout_pools,
            allow_nan=self.allow_nan)
        test_pool = LampleChartonCatalog.from_dict(
            set(test_keys),
            simplipy_engine=self.simplipy_engine,
            sample_strategy=self.sample_strategy,
            literal_prior=self.literal_prior,
            skeleton_codes={k: v for k, v in self.skeleton_codes.items() if k in test_keys},
            variables=self.variables,
            support_sampler_config=deepcopy(self.support_sampler_config),
            operator_weights=self.operator_weights,
            holdout_pools=self.holdout_pools,
            allow_nan=self.allow_nan)

        return train_pool, test_pool

    def __len__(self) -> int:
        '''
        Get the number of skeletons in the pool.

        Returns
        -------
        int
            The number of skeletons in the pool.
        '''
        return len(self.skeletons)


# Registry of generative catalog types (a mapping catalog spec selects one via its ``type`` key).
_GENERATIVE_CATALOGS: dict[str, type[GenerativeCatalog]] = {
    "lample_charton": LampleChartonCatalog,
}


def register_generative_catalog(name: str, cls: type[GenerativeCatalog]) -> None:
    """Register a custom :class:`GenerativeCatalog` under ``type: <name>`` so it can be built by config."""
    _GENERATIVE_CATALOGS[name] = cls


def is_open_generative_ref(spec: "str | Mapping[str, Any] | Catalog") -> bool:
    """True iff ``spec`` (a STRING ref / mapping) builds an OPEN generative catalog -- one that
    generates on the fly with NO fixed skeleton set.

    A generative spec carrying inline ``skeletons:`` is FROZEN (a fixed set), so it is NOT open: a
    consumer iterates it as a bounded set (each skeleton once), not as an unbounded stream. Declarative
    refs and ``.npz`` frozen catalogs are never open. Used by :class:`~symbolic_data.source.ProblemSource`
    to infer ``generate`` (unbounded) vs ``set`` (bounded) mode for a string catalog ref.
    """
    if isinstance(spec, Mapping):
        return spec.get("type") in _GENERATIVE_CATALOGS and not spec.get("skeletons")
    if isinstance(spec, Catalog):
        return False
    import yaml
    from symbolic_data.resolver import resolve
    try:
        artifact = resolve(spec)
        if artifact.path.endswith(".npz"):
            return False
        with open(artifact.path, encoding="utf-8") as handle:
            head = yaml.safe_load(handle)
    except Exception:  # noqa: BLE001 - unresolvable / non-yaml -> treat as not-open
        return False
    return isinstance(head, Mapping) and head.get("type") in _GENERATIVE_CATALOGS and not head.get("skeletons")


def build_catalog(spec: "str | Mapping[str, Any] | Catalog") -> Catalog:
    """Build the :class:`~symbolic_data.catalog.Catalog` a ``ProblemSource`` samples from.

    - a :class:`Catalog` instance -> returned as-is;
    - a mapping with a ``type`` key -> the registered :class:`GenerativeCatalog` built from it;
    - a string / path / ``name[@version]`` -> RESOLVED (local path or HF manifest), then dispatched on
      content: a generative spec (a ``type:`` yaml, optionally with inline frozen ``skeletons:``) builds
      the registered :class:`GenerativeCatalog`; anything else is a declarative
      :class:`~symbolic_data.catalog.ProblemCatalog`.
    """
    if isinstance(spec, Catalog):
        return spec
    if isinstance(spec, Mapping):
        cfg = dict(spec)
        ctype = cfg.pop("type", None)
        if ctype is None:
            raise ValueError("a mapping catalog spec must declare a 'type' (e.g. {'type': 'lample_charton', ...})")
        if ctype not in _GENERATIVE_CATALOGS:
            raise ValueError(f"unknown generative catalog type {ctype!r}; known: {sorted(_GENERATIVE_CATALOGS)}")
        return _GENERATIVE_CATALOGS[ctype].from_config(cfg)

    # string / path / name[@version]: resolve once, then peek the content to choose the catalog type.
    import yaml
    from symbolic_data.resolver import resolve
    artifact = resolve(spec)
    if not artifact.path.endswith(".npz"):
        try:
            with open(artifact.path, encoding="utf-8") as handle:
                head = yaml.safe_load(handle)
        except Exception:  # noqa: BLE001 - a non-yaml artifact is simply not a generative spec
            head = None
        if isinstance(head, Mapping) and head.get("type") in _GENERATIVE_CATALOGS:
            return _GENERATIVE_CATALOGS[head["type"]].from_config(artifact.path)
    return ProblemCatalog.load(spec)
