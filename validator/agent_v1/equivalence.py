"""Comparison-owned grouping; similarity never establishes scientific eligibility."""

from __future__ import annotations

import hashlib
from typing import Callable

from .comparison_models import CandidatePairEdge, ComparisonCandidate, EquivalenceGroup
from .pairing import _candidate_embeddings, _cosine


def propose_similarity_groups(
    candidates: list[ComparisonCandidate],
    *,
    embedding_provider: Callable | None = None,
    threshold: float = 0.90,
) -> list[list[str]]:
    """Deterministic complete-link proposals, avoiding similarity-chain closure.

    Missing/unavailable embeddings leave global comparison fully operational.
    """
    vectors = _candidate_embeddings(candidates, embedding_provider=embedding_provider)
    groups: list[list[str]] = []
    for candidate_id in sorted(vectors):
        for group in groups:
            if all(
                _cosine(vectors[candidate_id], vectors[other]) >= threshold
                for other in group
            ):
                group.append(candidate_id)
                break
        else:
            groups.append([candidate_id])
    return [group for group in groups if len(group) > 1]


def confirmed_equivalence_groups(
    *,
    paper_id: str,
    candidates: list[ComparisonCandidate],
    edges: list[CandidatePairEdge],
    explicit_groups: list[list[str]],
) -> list[EquivalenceGroup]:
    """Accept model-confirmed groups or unambiguous reference-anchored stars.

    Stars require every member to match the SAME anchor directly. Never infer a
    group from arbitrary connected components. Material incident edges prevent
    bypass so comparison disputes retain normal eligibility adjudication.
    """
    known = {candidate.candidate_id for candidate in candidates}
    blocked = {
        candidate_id
        for edge in edges
        if edge.relation != "semantic_equivalent"
        for candidate_id in (edge.left_candidate_id, edge.right_candidate_id)
    }
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        if edge.relation != "semantic_equivalent":
            continue
        left, right = edge.left_candidate_id, edge.right_candidate_id
        if left == right or left not in known or right not in known:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    proposals = [
        set(group)
        for group in sorted(explicit_groups, key=lambda group: tuple(sorted(group)))
    ]
    explicitly_grouped = set().union(*proposals) if proposals else set()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.origin != "bronze":
            continue
        anchor = candidate.candidate_id
        members = adjacency.get(anchor, set())
        # An ambiguous match to multiple reference anchors stays in adjudication.
        if (
            members
            and not ({anchor, *members} & explicitly_grouped)
            and all(adjacency.get(member) == {anchor} for member in members)
        ):
            proposals.append({anchor, *members})
    groups: list[EquivalenceGroup] = []
    occupied: set[str] = set()
    for members in sorted(proposals, key=lambda group: (-len(group), sorted(group))):
        if (
            len(members) < 2
            or not members.issubset(known)
            or members & (blocked | occupied)
        ):
            continue
        member_ids = sorted(members)
        digest = hashlib.sha256(repr((paper_id, member_ids)).encode()).hexdigest()[:20]
        groups.append(
            EquivalenceGroup(
                group_id=f"eq_{digest}",
                member_candidate_ids=member_ids,
                comparison_edge_ids=sorted(
                    edge.edge_id
                    for edge in edges
                    if edge.relation == "semantic_equivalent"
                    and {edge.left_candidate_id, edge.right_candidate_id}.issubset(
                        members
                    )
                ),
            )
        )
        occupied.update(members)
    return sorted(groups, key=lambda group: group.group_id)
