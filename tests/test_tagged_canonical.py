"""``tagged_canonical`` is simplify IN the tagged dialect -- a different normal form
from the prefix-dialect canonical, not a respelling of it. Specimens below are
measured engine output (acj-4, 2026-08-23), pinned."""
import pytest

from simplipy import SimpliPyEngine

from symbolic_data.token_ops import tagged_canonical


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4", install=True)


# infix -> the tagged canonical (measured, not derived)
SPECIMENS = [
    ("log(v1+1.4)+log(((v1)**(2))+1.3)",
     ["<add>", "log", "<add>", "v1", "<mul>", "7", "<div>", "5", "</mul>", "</add>",
      "log", "<add>", "pow", "v1", "2", "1.3", "</add>", "</add>"]),
    ("(1+2*v1)**(-3/2)",
     ["pow", "<add>", "<mul>", "2", "v1", "</mul>", "1", "</add>",
      "<mul>", "-3", "<div>", "2", "</mul>"]),
    ("exp(-(1+v1)**(-1))*(1+v1)**(-2)",
     ["<mul>", "exp", "neg", "inv", "<add>", "v1", "1", "</add>",
      "<div>", "pow", "<add>", "v1", "1", "</add>", "2", "</mul>"]),
    ("exp(-(v1+exp(-v1)))",
     ["exp", "<add>", "<sub>", "v1", "exp", "neg", "v1", "</add>"]),
]


@pytest.mark.parametrize("infix,expected", SPECIMENS, ids=[s[0] for s in SPECIMENS])
def test_tagged_canonical_is_simplify_in_the_tagged_dialect(engine: SimpliPyEngine, infix: str, expected: list[str]) -> None:
    assert tagged_canonical(engine, engine.read_infix(infix)) == expected


def test_converting_the_prefix_canonical_is_not_the_tagged_canonical(engine: SimpliPyEngine) -> None:
    # The design reason this helper exists: the two dialects canonicalize to different
    # normal forms (796 of 3,628 curated-catalog expressions differ). Here the prefix
    # canonical keeps ``neg`` OUTSIDE the sum; the tagged canonical distributes it
    # into a ``<sub>`` section.
    prefix = engine.read_infix("exp(-(v1+exp(-v1)))")
    converted = engine.to_tagged(engine.simplify(prefix))
    assert converted == ["exp", "neg", "<add>", "v1", "exp", "neg", "v1", "</add>"]
    assert tagged_canonical(engine, prefix) != converted
