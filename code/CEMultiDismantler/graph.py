import numpy as np
from GMM import GMM
import sys
from collections import defaultdict
import Mcc
import networkx as nx
import random
import inspect
try:
    import community as community_louvain  # python-louvain package
except Exception:
    community_louvain = None
try:
    import igraph as ig  # python-igraph
    import leidenalg  # Leiden community detection
except Exception:
    ig = None
    leidenalg = None

LEIDEN_NODE_THRESHOLD = 1000
_LEIDEN_FALLBACK_WARNED = False


def _warn_leiden_fallback_once():
    global _LEIDEN_FALLBACK_WARNED
    if _LEIDEN_FALLBACK_WARNED:
        return
    _LEIDEN_FALLBACK_WARNED = True
    print("CE-WARN: Leiden unavailable; falling back to Louvain. Install `python-igraph` + `leidenalg` to enable Leiden.")
class Graph:
    def __init__(self, N = 0):
        # N number of initialized nodes
        self.num_nodes = N
        self.adj_list = []
        self.edge_list = []
        self.num_edges = []
        self.max_rank = 0
        num_nodes_layer1 = N
        num_nodes_layer2 = N
        self.community_partition = [{}, {}]
        self.community_feat = [np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)]
        if N != 0:
            link1,link2 = GMM(N)
            G1 = nx.Graph()
            G1.add_nodes_from(range(N))
            G1.add_edges_from(link1)
            G2 = nx.Graph()
            G2.add_nodes_from(range(N))
            G2.add_edges_from(link2)
            self.adj_list = [list(G1.adjacency()),list(G2.adjacency())]
            self.edge_list = [G1.edges(),G2.edges()]
            self.num_edges = [len(self.edge_list[0]),len(self.edge_list[1])]
            self.max_rank = 0
            self.weights =[{},{}]
            self.ori_rank(G1,G2)
            self._calc_static_community_features(G1, G2)
            
    def reshape_graph(self, num_nodes, num_edges, edges_from, edges_to):
        self.num_nodes = num_nodes
        self.num_edges.append(num_edges)
        adj_list = [[] for _ in range(num_nodes)]
        edge_list = [(edges_from[i], edges_to[i]) for i in range(num_edges)]
        for i in range(num_edges):
            x, y = edges_from[i], edges_to[i]
            adj_list[x].append(y)
            adj_list[y].append(x)
        self.adj_list.append(adj_list)
        self.edge_list.append(edge_list)

    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)

    def _run_louvain(self, G_nx: nx.Graph):
        if community_louvain is None:
            return {}

        seed = 0
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            try:
                sig = inspect.signature(community_louvain.best_partition)
                if "random_state" in sig.parameters:
                    return community_louvain.best_partition(G_nx, random_state=seed)
            except Exception:
                pass

            # Older python-louvain: isolate RNG so community detection doesn't perturb training RNG streams.
            random.seed(seed)
            np.random.seed(seed)
            return community_louvain.best_partition(G_nx)
        except Exception:
            return {}
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _run_leiden(self, G_nx: nx.Graph):
        if ig is None or leidenalg is None:
            return {}

        seed = 0
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            nodes = list(G_nx.nodes())
            if not nodes:
                return {}
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
            g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

            # Isolate RNG to keep training sampling reproducible.
            random.seed(seed)
            np.random.seed(seed)

            try:
                sig = inspect.signature(leidenalg.find_partition)
                if "seed" in sig.parameters:
                    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, seed=seed)
                else:
                    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
            except Exception:
                part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)

            membership = getattr(part, "membership", None)
            if membership is None:
                return {}
            return {node: int(membership[idx]) for node, idx in node_to_idx.items()}
        except Exception:
            return {}
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _run_partition(self, G_nx: nx.Graph):
        if self.num_nodes and self.num_nodes < LEIDEN_NODE_THRESHOLD:
            if ig is None or leidenalg is None:
                _warn_leiden_fallback_once()
            else:
                part = self._run_leiden(G_nx)
                if part:
                    return part
        return self._run_louvain(G_nx)

    def _layer_z_p(self, G_nx: nx.Graph, partition):
        n = self.num_nodes
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        part = {i: int(partition.get(i, 0)) for i in range(n)}
        within_degree = np.zeros(n, dtype=np.float32)
        participation = np.zeros(n, dtype=np.float32)

        comm_to_within = defaultdict(list)
        for v in range(n):
            v_comm = part[v]
            neighbors = list(G_nx.neighbors(v))
            k_total = len(neighbors)
            if k_total == 0:
                within_degree[v] = 0.0
                participation[v] = 0.0
                comm_to_within[v_comm].append(0.0)
                continue

            comm_counts = defaultdict(int)
            for nb in neighbors:
                comm_counts[part.get(nb, 0)] += 1

            k_in = float(comm_counts.get(v_comm, 0))
            within_degree[v] = k_in
            participation[v] = 1.0 - sum((cnt / k_total) ** 2 for cnt in comm_counts.values())
            comm_to_within[v_comm].append(k_in)

        comm_stats = {}
        for comm, vals in comm_to_within.items():
            arr = np.asarray(vals, dtype=np.float32)
            comm_stats[comm] = (float(arr.mean()) if arr.size else 0.0, float(arr.std()) if arr.size else 0.0)

        z = np.zeros(n, dtype=np.float32)
        for v in range(n):
            v_comm = part[v]
            mean_c, std_c = comm_stats.get(v_comm, (0.0, 0.0))
            z[v] = (within_degree[v] - mean_c) / std_c if std_c > 0 else 0.0
        return z, participation

    def _zscore(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        std = float(arr.std()) if arr.size else 0.0
        if std <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        mean = float(arr.mean())
        return (arr - mean) / std

    def _minmax(self, arr: np.ndarray, vmin: float, vmax: float):
        arr = np.asarray(arr, dtype=np.float32)
        denom = float(vmax - vmin)
        if denom <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - float(vmin)) / denom

    def _calc_static_community_features(self, G1: nx.Graph, G2: nx.Graph):
        if self.num_nodes == 0:
            self.community_partition = [{}, {}]
            self.community_feat = [np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)]
            return

        graphs = [G1, G2]
        partitions = []
        z_raw = []
        p_raw = []
        for layer, G in enumerate(graphs):
            part = self._run_partition(G)
            if not part:
                part = {i: 0 for i in range(self.num_nodes)}
            partitions.append({i: int(part.get(i, 0)) for i in range(self.num_nodes)})
            z_l, p_l = self._layer_z_p(G, partitions[layer])
            z_raw.append(z_l)
            p_raw.append(p_l)

        # Layer-wise z-score standardization
        z_std = [self._zscore(z_raw[0]), self._zscore(z_raw[1])]
        p_std = [self._zscore(p_raw[0]), self._zscore(p_raw[1])]

        # Global Min-Max across (layer, feature) after standardization
        all_vals = np.concatenate([z_std[0], p_std[0], z_std[1], p_std[1]]).astype(np.float32, copy=False)
        vmin = float(all_vals.min()) if all_vals.size else 0.0
        vmax = float(all_vals.max()) if all_vals.size else 0.0

        self.community_partition = partitions
        self.community_feat = [
            np.stack([self._minmax(z_std[0], vmin, vmax), self._minmax(p_std[0], vmin, vmax)], axis=1),
            np.stack([self._minmax(z_std[1], vmin, vmax), self._minmax(p_std[1], vmin, vmax)], axis=1),
        ]
        
class GSet:
    def __init__(self):
        self.graph_pool = {}

    def InsertGraph(self, gid, graph):
        assert gid not in self.graph_pool
        self.graph_pool[gid] = graph

    def Sample(self):
        assert self.graph_pool
        gid = np.random.choice(list(self.graph_pool.keys()))
        return self.graph_pool[gid]

    def Get(self, gid):
        assert gid in self.graph_pool
        return self.graph_pool[gid]

    def Clear(self):
        self.graph_pool.clear()
        
class Graph_test:
    def __init__(self,G1,G2):
        
        self.num_nodes = len(G1.nodes)
        self.adj_list = [list(G1.adjacency()),list(G2.adjacency())]
        self.edge_list = [G1.edges(),G2.edges()]
        self.num_edges = [len(self.edge_list[0]),len(self.edge_list[1])]
        self.max_rank = 0
        self.weights =[{},{}]
        self.community_partition = [{}, {}]
        self.community_feat = [np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)]
        self.ori_rank(G1,G2)
        self._calc_static_community_features(G1, G2)
        
    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        return G1, G2

    def _run_louvain(self, G_nx: nx.Graph):
        if community_louvain is None:
            return {}

        seed = 0
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            try:
                sig = inspect.signature(community_louvain.best_partition)
                if "random_state" in sig.parameters:
                    return community_louvain.best_partition(G_nx, random_state=seed)
            except Exception:
                pass

            random.seed(seed)
            np.random.seed(seed)
            return community_louvain.best_partition(G_nx)
        except Exception:
            return {}
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _run_leiden(self, G_nx: nx.Graph):
        if ig is None or leidenalg is None:
            return {}

        seed = 0
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            nodes = list(G_nx.nodes())
            if not nodes:
                return {}
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
            g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

            random.seed(seed)
            np.random.seed(seed)

            try:
                sig = inspect.signature(leidenalg.find_partition)
                if "seed" in sig.parameters:
                    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, seed=seed)
                else:
                    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
            except Exception:
                part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)

            membership = getattr(part, "membership", None)
            if membership is None:
                return {}
            return {node: int(membership[idx]) for node, idx in node_to_idx.items()}
        except Exception:
            return {}
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _run_partition(self, G_nx: nx.Graph):
        if self.num_nodes and self.num_nodes < LEIDEN_NODE_THRESHOLD:
            if ig is None or leidenalg is None:
                _warn_leiden_fallback_once()
            else:
                part = self._run_leiden(G_nx)
                if part:
                    return part
        return self._run_louvain(G_nx)

    def _layer_z_p(self, G_nx: nx.Graph, partition):
        n = self.num_nodes
        if n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        part = {i: int(partition.get(i, 0)) for i in range(n)}
        within_degree = np.zeros(n, dtype=np.float32)
        participation = np.zeros(n, dtype=np.float32)

        comm_to_within = defaultdict(list)
        for v in range(n):
            v_comm = part[v]
            neighbors = list(G_nx.neighbors(v))
            k_total = len(neighbors)
            if k_total == 0:
                within_degree[v] = 0.0
                participation[v] = 0.0
                comm_to_within[v_comm].append(0.0)
                continue

            comm_counts = defaultdict(int)
            for nb in neighbors:
                comm_counts[part.get(nb, 0)] += 1

            k_in = float(comm_counts.get(v_comm, 0))
            within_degree[v] = k_in
            participation[v] = 1.0 - sum((cnt / k_total) ** 2 for cnt in comm_counts.values())
            comm_to_within[v_comm].append(k_in)

        comm_stats = {}
        for comm, vals in comm_to_within.items():
            arr = np.asarray(vals, dtype=np.float32)
            comm_stats[comm] = (float(arr.mean()) if arr.size else 0.0, float(arr.std()) if arr.size else 0.0)

        z = np.zeros(n, dtype=np.float32)
        for v in range(n):
            v_comm = part[v]
            mean_c, std_c = comm_stats.get(v_comm, (0.0, 0.0))
            z[v] = (within_degree[v] - mean_c) / std_c if std_c > 0 else 0.0
        return z, participation

    def _zscore(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        std = float(arr.std()) if arr.size else 0.0
        if std <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        mean = float(arr.mean())
        return (arr - mean) / std

    def _minmax(self, arr: np.ndarray, vmin: float, vmax: float):
        arr = np.asarray(arr, dtype=np.float32)
        denom = float(vmax - vmin)
        if denom <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - float(vmin)) / denom

    def _calc_static_community_features(self, G1: nx.Graph, G2: nx.Graph):
        if self.num_nodes == 0:
            self.community_partition = [{}, {}]
            self.community_feat = [np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)]
            return

        graphs = [G1, G2]
        partitions = []
        z_raw = []
        p_raw = []
        for layer, G in enumerate(graphs):
            part = self._run_partition(G)
            if not part:
                part = {i: 0 for i in range(self.num_nodes)}
            partitions.append({i: int(part.get(i, 0)) for i in range(self.num_nodes)})
            z_l, p_l = self._layer_z_p(G, partitions[layer])
            z_raw.append(z_l)
            p_raw.append(p_l)

        z_std = [self._zscore(z_raw[0]), self._zscore(z_raw[1])]
        p_std = [self._zscore(p_raw[0]), self._zscore(p_raw[1])]

        all_vals = np.concatenate([z_std[0], p_std[0], z_std[1], p_std[1]]).astype(np.float32, copy=False)
        vmin = float(all_vals.min()) if all_vals.size else 0.0
        vmax = float(all_vals.max()) if all_vals.size else 0.0

        self.community_partition = partitions
        self.community_feat = [
            np.stack([self._minmax(z_std[0], vmin, vmax), self._minmax(p_std[0], vmin, vmax)], axis=1),
            np.stack([self._minmax(z_std[1], vmin, vmax), self._minmax(p_std[1], vmin, vmax)], axis=1),
        ]
    
