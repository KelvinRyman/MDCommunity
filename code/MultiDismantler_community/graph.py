import numpy as np
from GMM import GMM
import sys
from collections import defaultdict
import Mcc
import networkx as nx
try:
    from networkx.algorithms.community import louvain_communities
except Exception:
    louvain_communities = None
try:
    import community as community_louvain  # python-louvain package
except Exception:
    community_louvain = None
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
            self.community_partition = {}
            self.node_participation = []
            self.node_zscore = []
            self.ori_rank(G1,G2)
            self._calc_static_community_features()
        else:
            self.community_partition = {}
            self.node_participation = []
            self.node_zscore = []
            
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
    
    def _calc_static_community_features(self):
        """Compute community-aware static features: participation coefficient P and within-module z-score."""
        if self.num_nodes == 0:
            self.community_partition = {}
            self.node_participation = []
            self.node_zscore = []
            return

        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(self.num_nodes))
        for layer_adj in self.adj_list:
            for i, neighbors in layer_adj:
                for j in neighbors:
                    if i < j and not G_nx.has_edge(i, j):
                        G_nx.add_edge(i, j)

        partition = self._run_louvain(G_nx)
        if not partition:
            # fallback: single community
            partition = {i: 0 for i in range(self.num_nodes)}
        self.community_partition = partition.copy()

        # precompute degrees by community
        comm_degrees = defaultdict(list)
        node_degree = dict(G_nx.degree())
        # participation: k_i^in per community
        for v in range(self.num_nodes):
            v_comm = partition.get(v, 0)
            neighbors = list(G_nx.neighbors(v))
            k_total = len(neighbors)
            # count edges to each community
            comm_counts = defaultdict(int)
            for n in neighbors:
                comm_counts[partition.get(n, 0)] += 1
            k_in = comm_counts[v_comm]
            comm_degrees[v_comm].append(k_in)
        # compute mean/std per community for k_in
        comm_stats = {}
        for comm, vals in comm_degrees.items():
            arr = np.array(vals, dtype=np.float32)
            mean = float(arr.mean()) if len(arr) > 0 else 0.0
            std = float(arr.std()) if len(arr) > 0 else 0.0
            comm_stats[comm] = (mean, std)

        eps = 1e-8
        participation = np.zeros(self.num_nodes, dtype=np.float32)
        zscore = np.zeros(self.num_nodes, dtype=np.float32)
        for v in range(self.num_nodes):
            v_comm = partition.get(v, 0)
            neighbors = list(G_nx.neighbors(v))
            k_total = len(neighbors)
            if k_total == 0:
                participation[v] = 0.0
                zscore[v] = 0.0
                continue
            comm_counts = defaultdict(int)
            for n in neighbors:
                comm_counts[partition.get(n, 0)] += 1
            # Participation coefficient: P_i = 1 - sum_s (k_is/k_i)^2
            participation[v] = 1.0 - sum((cnt / k_total) ** 2 for cnt in comm_counts.values())
            k_in = comm_counts[v_comm]
            mean_c, std_c = comm_stats.get(v_comm, (0.0, 0.0))
            # within-module z-score: z_i = (k_in - mean_c) / std_c
            zscore[v] = (k_in - mean_c) / std_c if std_c > 0 else 0.0

        self.node_participation = participation
        self.node_zscore = zscore

    def _run_louvain(self, G_nx):
        """Run Louvain if available; fallback to python-louvain; otherwise return empty mapping."""
        try:
            if louvain_communities is not None:
                comms = louvain_communities(G_nx, seed=0)
                return {node: idx for idx, c in enumerate(comms) for node in c}
            if community_louvain is not None:
                partition = community_louvain.best_partition(G_nx)
                return partition
        except Exception:
            pass
        return {}
        
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
        self.ori_rank(G1,G2)
        self.community_partition = {}
        self.node_participation = []
        self.node_zscore = []
        self._compute_community_features(G1, G2)
        
    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        return G1, G2
    
    def _compute_community_features(self, G1, G2):
        """Compute participation and z-score for test graphs using merged layers."""
        if self.num_nodes == 0:
            self.community_partition = {}
            self.node_participation = []
            self.node_zscore = []
            return
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(self.num_nodes))
        G_nx.add_edges_from(G1.edges())
        G_nx.add_edges_from(G2.edges())

        partition = None
        try:
            if louvain_communities is not None:
                comms = louvain_communities(G_nx, seed=0)
                partition = {node: idx for idx, c in enumerate(comms) for node in c}
            elif community_louvain is not None:
                partition = community_louvain.best_partition(G_nx)
        except Exception:
            partition = None

        if not partition:
            partition = {i: 0 for i in range(self.num_nodes)}
        self.community_partition = partition.copy()

        comm_degrees = defaultdict(list)
        for v in range(self.num_nodes):
            v_comm = partition.get(v, 0)
            neighbors = list(G_nx.neighbors(v))
            comm_counts = defaultdict(int)
            for n in neighbors:
                comm_counts[partition.get(n, 0)] += 1
            k_in = comm_counts[v_comm]
            comm_degrees[v_comm].append(k_in)

        comm_stats = {}
        for comm, vals in comm_degrees.items():
            arr = np.array(vals, dtype=np.float32)
            mean = float(arr.mean()) if len(arr) > 0 else 0.0
            std = float(arr.std()) if len(arr) > 0 else 0.0
            comm_stats[comm] = (mean, std)

        participation = np.zeros(self.num_nodes, dtype=np.float32)
        zscore = np.zeros(self.num_nodes, dtype=np.float32)
        for v in range(self.num_nodes):
            v_comm = partition.get(v, 0)
            neighbors = list(G_nx.neighbors(v))
            k_total = len(neighbors)
            if k_total == 0:
                participation[v] = 0.0
                zscore[v] = 0.0
                continue
            comm_counts = defaultdict(int)
            for n in neighbors:
                comm_counts[partition.get(n, 0)] += 1
            participation[v] = 1.0 - sum((cnt / k_total) ** 2 for cnt in comm_counts.values())
            k_in = comm_counts[v_comm]
            mean_c, std_c = comm_stats.get(v_comm, (0.0, 0.0))
            zscore[v] = (k_in - mean_c) / std_c if std_c > 0 else 0.0

        self.node_participation = participation
        self.node_zscore = zscore
    
