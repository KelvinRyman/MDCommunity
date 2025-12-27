# mvc_env.py
from __future__ import annotations

from typing import List
import random

from torch import normal

from graph import Graph
import networkx as nx
import Mcc


class MvcEnv:
    def __init__(self, norm: int):
        self.norm = norm

        self.graph = Graph(0)
        self.numCoveredEdges = [0, 0]
        self.CcNum = 1.0
        self.state_seq: List[List[int]] = []
        self.act_seq: List[int] = []
        self.action_list: List[int] = []
        self.reward_seq: List[float] = []
        self.sum_rewards: List[float] = []
        self.covered_set = set()
        self.avail_list: List[int] = []
        self.remove_edge = [set(), set()]
        self.state_seq_edges = []
        self.MaxCCList = [1]
        self.score = 0.0

        self.flag = 0
        self.G1 = None
        self.G2 = None

        # Scheme 1 reward shaping (credit assignment): previous LMCC size before action.
        self.prev_rank = 0.0

    def s0(self, _g: Graph):
        self.graph = _g
        self.covered_set.clear()
        self.action_list.clear()
        self.numCoveredEdges = [0, 0]
        self.CcNum = 1.0
        self.state_seq.clear()
        self.act_seq.clear()
        self.reward_seq.clear()
        self.sum_rewards.clear()
        self.remove_edge[0].clear()
        self.remove_edge[1].clear()
        self.state_seq_edges.clear()
        self.MaxCCList = [1]
        self.score = 0.0
        self.flag = 0
        self.G1 = None
        self.G2 = None
        # Initialize cached NetworkX graphs and set initial LMCC size for drop-based reward.
        self.prev_rank = float(self.getMaxConnectedNodesNum())

    def _boundary_candidates(self) -> List[int]:
        if not hasattr(self.graph, "boundary_nodes") or not self.graph.boundary_nodes:
            return []
        available = set(self._available_nodes())
        return [n for n in self.graph.boundary_nodes if n not in self.covered_set and n in available]

    def _available_nodes(self) -> List[int]:
        assert self.graph
        avail = []
        for i in range(self.graph.num_nodes):
            if i in self.covered_set:
                continue
            useful1 = any(
                neigh not in self.covered_set and (i, neigh) not in self.remove_edge[0]
                for neigh in self.graph.adj_list[0][i][1]
            )
            useful2 = any(
                neigh not in self.covered_set and (i, neigh) not in self.remove_edge[1]
                for neigh in self.graph.adj_list[1][i][1]
            )
            if useful1 and useful2:
                avail.append(i)
        return avail

    def getValidActions(self) -> List[int]:
        """
        Divide-and-Conquer Action Pruning:
        - Phase A: if any boundary nodes remain, only allow boundary actions.
        - Phase B: if boundary nodes exhausted, allow all remaining valid nodes.
        """
        available = self._available_nodes()
        if not available:
            return []
        available_set = set(available)
        boundary = (
            [n for n in getattr(self.graph, "boundary_nodes", set()) if n not in self.covered_set and n in available_set]
            if getattr(self.graph, "boundary_nodes", None)
            else []
        )
        return boundary if boundary else available

    def step(self, a: int) -> float:
        assert self.graph
        assert a not in self.covered_set

        self.state_seq.append(self.action_list.copy())
        remove_edge = [self.remove_edge[0].copy(), self.remove_edge[1].copy()]
        self.state_seq_edges.append(remove_edge)
        self.act_seq.append(a)

        self.covered_set.add(a)
        self.action_list.append(a)

        for i in range(2):
            for neigh in self.graph.adj_list[i][a][1]:
                if neigh not in self.covered_set and (neigh, a) not in self.remove_edge[i]:
                    self.numCoveredEdges[i] += 1

        r_t = self.getReward(a)
        self.reward_seq.append(r_t)
        self.sum_rewards.append(r_t)
        return r_t

    def stepWithoutReward(self, a: int):
        assert self.graph
        assert a not in self.covered_set
        self.covered_set.add(a)
        self.action_list.append(a)

        for i in range(2):
            for neigh in list(self.graph.adj_list[i][a][1]):
                if neigh not in self.covered_set and (neigh, a) not in self.remove_edge[i]:
                    self.numCoveredEdges[i] += 1

        # Evaluation objective stays identical to unit-cost (AUDC-like), independent of shaping.
        r_obj = self.getObjectiveReward(a)
        self.score += -1.0 * r_obj
        self.MaxCCList.append(-1.0 * r_obj * float(self.graph.num_nodes))

    def randomAction(self) -> int:
        candidates = self.getValidActions()
        assert candidates
        return random.choice(candidates)

    def isTerminal(self) -> bool:
        assert self.graph
        return self.graph.num_edges[0] == (self.numCoveredEdges[0] + len(self.remove_edge[0]) / 2) or self.graph.num_edges[
            1
        ] == (self.numCoveredEdges[1] + len(self.remove_edge[1]) / 2)

    def getReward(self, a: int) -> float:
        """
        Scheme 1 reward (Time-Weighted Drop Reward):
        - Credit assignment based on LMCC drop caused by action `a`.
        - Time-weighted to emphasize early dismantling.
        """
        orig_node_num = float(self.graph.num_nodes)

        rank = self.getMaxConnectedNodesNum(a)
        base_penalty = -float(rank) / (self.graph.max_rank * orig_node_num)

        max_steps = orig_node_num
        current_step = float(len(self.action_list))
        steps_remaining = max(0.0, max_steps - current_step)
        time_ratio = steps_remaining / max_steps
        lambda_coef = 1.0
        time_weight = 1.0 + (lambda_coef * time_ratio)
        return float(base_penalty * time_weight)

    def getObjectiveReward(self, a: int) -> float:
        """
        Unit-cost objective reward (identical to original MultiDismantler_unit_cost):
          r = - rank_t / (max_rank * N)
        Used for evaluation (vc/AUDC) and logging.
        """
        orig_node_num = float(self.graph.num_nodes) if self.graph.num_nodes > 0 else 1.0
        rank = float(self.getMaxConnectedNodesNum(a))
        return -rank / (float(self.graph.max_rank) * orig_node_num)
    

    def getMaxConnectedNodesNum(self, a=None) -> float:
        assert self.graph
        if self.flag == 0:
            self.G1 = nx.Graph()
            self.G2 = nx.Graph()
            self.G1.add_nodes_from(range(0, self.graph.num_nodes))
            self.G2.add_nodes_from(range(0, self.graph.num_nodes))
            for i, neighbors in self.graph.adj_list[0]:
                for j in neighbors:
                    if i not in self.covered_set and j not in self.covered_set and (i, j) not in self.remove_edge[0]:
                        self.G1.add_edge(i, j)
            for i, neighbors in self.graph.adj_list[1]:
                for j in neighbors:
                    if i not in self.covered_set and j not in self.covered_set and (i, j) not in self.remove_edge[1]:
                        self.G2.add_edge(i, j)
            self.flag = 1
        else:
            self.G1.remove_node(a)
            self.G2.remove_node(a)
        connected_components = Mcc.MCC(self.G1, self.G2, self.remove_edge)
        rank = Mcc.find_max_set_length(connected_components)
        return float(rank)
