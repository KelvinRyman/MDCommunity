from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import inspect
import os
import random

import numpy as np
import networkx as nx

try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None


@dataclass(frozen=True)
class CommunityPrior:
    boundary_nodes: Set[int]
    node_feature: np.ndarray  # shape [N], float32 in [0,1]


def _isolate_rng(seed: int = 0):
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    return py_state, np_state


def _restore_rng(py_state, np_state):
    random.setstate(py_state)
    np.random.set_state(np_state)


def louvain_partition(G: nx.Graph, seed: int = 0) -> Dict[int, int]:
    if community_louvain is None:
        raise RuntimeError("python-louvain is not installed (import community failed).")

    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        try:
            sig = inspect.signature(community_louvain.best_partition)
            if "random_state" in sig.parameters:
                return community_louvain.best_partition(G, random_state=seed)
        except Exception:
            pass

        # Old python-louvain: keep deterministic and avoid polluting global RNG used by training.
        random.seed(seed)
        np.random.seed(seed)
        return community_louvain.best_partition(G)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def participation_and_boundary(G: nx.Graph, part: Dict[int, int], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - participation coefficient P in [0,1], shape [n]
    - boundary flag in {0,1}, shape [n]
    """
    P = np.zeros(n, dtype=np.float32)
    boundary = np.zeros(n, dtype=np.float32)
    for u in range(n):
        u_comm = int(part.get(u, 0))
        neighbors = list(G.neighbors(u))
        k_total = len(neighbors)
        if k_total <= 0:
            P[u] = 0.0
            boundary[u] = 0.0
            continue

        comm_counts: Dict[int, int] = {}
        is_b = False
        for v in neighbors:
            v_comm = int(part.get(v, 0))
            comm_counts[v_comm] = comm_counts.get(v_comm, 0) + 1
            if v_comm != u_comm:
                is_b = True

        P[u] = 1.0 - sum((cnt / k_total) ** 2 for cnt in comm_counts.values())
        boundary[u] = 1.0 if is_b else 0.0
    return P, boundary


def compute_prior_feature(
    G: nx.Graph,
    *,
    feature: str = "boundary",
    seed: int = 0,
) -> CommunityPrior:
    """
    feature:
      - 'boundary': binary 0/1
      - 'participation': continuous in [0,1]
    """
    n = G.number_of_nodes()
    if n <= 0:
        return CommunityPrior(boundary_nodes=set(), node_feature=np.zeros((0,), dtype=np.float32))

    if feature == "none":
        return CommunityPrior(boundary_nodes=set(), node_feature=np.zeros((n,), dtype=np.float32))

    part = louvain_partition(G, seed=seed)
    P, boundary = participation_and_boundary(G, part, n=n)

    if feature == "participation":
        feat = P
    elif feature == "boundary":
        feat = boundary
    else:
        raise ValueError("feature must be one of: 'none', 'boundary', 'participation'")

    boundary_nodes = set(np.where(boundary > 0.5)[0].tolist())
    feat = np.asarray(feat, dtype=np.float32)
    # Keep it in [0,1] and finite.
    feat = np.clip(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    return CommunityPrior(boundary_nodes=boundary_nodes, node_feature=feat)


def apply_prior_to_two_layer_graph(
    *,
    G1: nx.Graph,
    G2: nx.Graph,
    feature: str = "boundary",
    seed: int = 0,
) -> Tuple[List[np.ndarray], Set[int]]:
    """
    Returns (per-layer node_feature list, union boundary nodes).
    """
    n = G1.number_of_nodes()
    if feature == "none":
        zeros = np.zeros((n,), dtype=np.float32)
        return [zeros.copy(), zeros.copy()], set()
    p1 = compute_prior_feature(G1, feature=feature, seed=seed)
    p2 = compute_prior_feature(G2, feature=feature, seed=seed)
    if p1.node_feature.shape[0] != n:
        p1_feat = np.zeros((n,), dtype=np.float32)
    else:
        p1_feat = p1.node_feature
    if p2.node_feature.shape[0] != n:
        p2_feat = np.zeros((n,), dtype=np.float32)
    else:
        p2_feat = p2.node_feature
    boundary_union = set(p1.boundary_nodes) | set(p2.boundary_nodes)
    return [p1_feat, p2_feat], boundary_union


def cache_path_for_real(dataset_name: str, layers: Tuple[int, int], feature: str) -> str:
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")
    return os.path.join("..", "..", "data", "real_cache", f"{safe_name}_layers{layers[0]}{layers[1]}_{feature}.npz")
