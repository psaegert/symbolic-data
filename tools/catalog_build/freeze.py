"""The 1,024-problem subsamples (srbf rulings 2026-10-04/05): a subset with more than 1,024 problems is subsampled
ONCE, and the frozen list is the only thing srbf evaluates. No random-number generator, so it reproduces anywhere:
each problem is ranked by sha256("srbf-1024:<set>:<subset>:<problem id>"), lowest first; the places are split over the
authors' own groups in proportion to their sizes (largest remainder); within a group the lowest-ranked are taken."""
import hashlib
from collections import defaultdict

K = 1024


def allocate(sizes, k):
    """largest-remainder apportionment of k places over groups of the given sizes (ties: group name)"""
    n = sum(sizes.values())
    quota = {g: k * s / n for g, s in sizes.items()}
    take = {g: int(q) for g, q in quota.items()}
    for g in sorted(sizes, key=lambda g: (-(quota[g] - take[g]), g))[: k - sum(take.values())]:
        take[g] += 1
    return take


def draw(set_id, subset, members, k=K):
    """members: [(problem id, group)] -> the frozen ids, in rank order within each group"""
    salt = f"srbf-1024:{set_id}:{subset}"
    rank = lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()  # noqa: E731
    groups = defaultdict(list)
    for i, g in members:
        groups[g].append(i)
    assert len({i for i, _ in members}) == len(members), f"{set_id}: duplicate problem ids"
    take = allocate({g: len(v) for g, v in groups.items()}, k)
    ids = [i for g in sorted(groups) for i in sorted(groups[g], key=rank)[: take[g]]]
    return salt, take, {g: len(v) for g, v in groups.items()}, ids
