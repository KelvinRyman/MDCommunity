# mvc_env.py
from typing import List, Set
import random
from disjoint_set import DisjointSet
from graph import Graph
import networkx as nx
import numpy as np
import Mcc
import math
from cdlib import algorithms

class MvcEnv:
    def __init__(self, norm):
        # Constructor method of the MvcEnv class that initializes the instance variables
        self.norm = norm  
        self.graph = Graph(0)
        self.numCoveredEdges = [0,0]  # of edges covered
        self.CcNum = 1.0  # connected component
        self.state_seq = []  # Storing State Sequences
        self.act_seq = []  # the action 
        self.action_list = []  # the action list 
        self.reward_seq = []  # the reward list
        self.sum_rewards = []  # the accumulatioin reward
        self.covered_set = set()  # the set of covered nodes
        self.avail_list = []  # the list of available nodes
        #self.single_nodes_after_act = set()
        #self.not_chose_nodes = set()
        self.remove_edge = [set(),set()]
        self.state_seq_edges = []
        self.MaxCCList = [1]
        self.score = 0.0
        self.flag = 0
        self.G1 = None
        self.G2 = None
        # HCA Hyperparameters
        self.beta = 0.5  # Weight for R_Bridge
        self.tau = 0.5   # Threshold for f_het

    def calculate_hca_features(self):
        """
        Calculate HCA-Dismantler features: 
        1. f_het (Cross-Layer Heterogeneity Score)
        2. f_impact (Potential Cascade Impact)
        3. f_roi (Cost-Damage ROI Index)
        And store them in self.graph.node_features
        """
        # Ensure NX graphs are built
        if self.G1 is None or self.G2 is None:
            self.G1 = nx.Graph()
            self.G2 = nx.Graph()
            self.G1.add_nodes_from(range(0, self.graph.num_nodes))
            self.G2.add_nodes_from(range(0, self.graph.num_nodes))
            for i, neighbors in self.graph.adj_list[0]:
                for j in neighbors:
                    self.G1.add_edge(i, j)
            for i, neighbors in self.graph.adj_list[1]:
                for j in neighbors:
                    self.G2.add_edge(i, j)
        
        # 1. Community Detection (Leiden)
        try:
            # cdlib input is networkx graph
            coms1 = algorithms.leiden(self.G1)
            coms2 = algorithms.leiden(self.G2)
            
            # Store communities in Graph object for PrepareBatchGraph
            if self.graph.subgraphs is None:
                self.graph.subgraphs = [[], []]
            self.graph.subgraphs[0] = coms1.communities
            self.graph.subgraphs[1] = coms2.communities

            # Create node -> community ID map
            node2com1 = {}
            node2com2 = {}
            com_nodes1 = {} # id -> set of nodes
            com_nodes2 = {}
            
            for cid, nodes in enumerate(coms1.communities):
                s_nodes = set(nodes)
                com_nodes1[cid] = s_nodes
                for u in nodes:
                    node2com1[u] = cid
                    
            for cid, nodes in enumerate(coms2.communities):
                s_nodes = set(nodes)
                com_nodes2[cid] = s_nodes
                for u in nodes:
                    node2com2[u] = cid
        except Exception as e:
            # Fallback if leiden fails
            print(f"Warning: Community detection failed ({e}). Using dummy features.")
            self.graph.node_features = np.zeros((self.graph.num_nodes, 3))
            self.node2com1 = {u:0 for u in range(self.graph.num_nodes)}
            self.node2com2 = {u:0 for u in range(self.graph.num_nodes)}
            return

        self.node2com1 = node2com1
        self.node2com2 = node2com2
        
        # 2. Calculate Features
        features = np.zeros((self.graph.num_nodes, 3))
        epsilon = 1e-6
        
        for u in range(self.graph.num_nodes):
            cid1 = node2com1.get(u, -1)
            cid2 = node2com2.get(u, -1)
            
            if cid1 == -1 or cid2 == -1:
                continue
                
            set1 = com_nodes1[cid1]
            set2 = com_nodes2[cid2]
            
            # f_het: 1 - Jaccard
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard = intersection / (union + epsilon)
            f_het = 1.0 - jaccard
            
            # f_impact: log(|C1|) * log(|C2|)
            size1 = len(set1)
            size2 = len(set2)
            f_impact = math.log(size1 + 1) * math.log(size2 + 1)
            
            # Cost(u): Degree Cost (Sum of degrees in both layers)
            deg1 = self.G1.degree[u]
            deg2 = self.G2.degree[u]
            cost_u = deg1 + deg2
            
            # f_roi
            f_roi = (f_het * f_impact) / (cost_u + epsilon)
            
            features[u] = [f_het, f_impact, f_roi]
            
        self.graph.node_features = features
        self.initial_inter_community_edges = self.count_inter_community_edges()

    def count_inter_community_edges(self):
        """Count total edges that connect different communities in current state"""
        count = 0
        # Check Layer 1
        for u, v in self.G1.edges():
            if u in self.covered_set or v in self.covered_set: continue
            if (u, v) in self.remove_edge[0]: continue
            if self.node2com1.get(u) != self.node2com1.get(v):
                count += 1
        # Check Layer 2
        for u, v in self.G2.edges():
            if u in self.covered_set or v in self.covered_set: continue
            if (u, v) in self.remove_edge[1]: continue
            if self.node2com2.get(u) != self.node2com2.get(v):
                count += 1
        return count

    def s0(self, _g: Graph):
        # reset the state of the environment
        self.graph = _g  
        self.covered_set.clear()  
        self.action_list.clear()  
        self.numCoveredEdges = [0,0]
        self.CcNum = 1.0  
        self.state_seq.clear()  
        self.act_seq.clear()  
        self.reward_seq.clear()  
        self.sum_rewards.clear() 
        #self.single_nodes_after_act.clear()
        #self.not_chose_nodes.clear()
        self.remove_edge[0].clear()
        self.remove_edge[1].clear()
        self.state_seq_edges.clear()
        self.MaxCCList = [1]
        self.score = 0.0
        self.flag = 0
        self.G1 = None
        self.G2 = None
        self.calculate_hca_features() # Build feats and G1/G2
        self.getMaxConnectedNodesNum()
    def step(self, a):
       
        assert self.graph
        assert a not in self.covered_set

        self.state_seq.append(self.action_list.copy()) 
        remove_edge = [self.remove_edge[0].copy(),self.remove_edge[1].copy()]
        self.state_seq_edges.append(remove_edge)
        self.act_seq.append(a)  
        self.covered_set.add(a)  
        self.action_list.append(a)  
        for i in range(2):
            for neigh in self.graph.adj_list[i][a][1]:              
                    if neigh not in self.covered_set and (neigh,a) not in self.remove_edge[i]:
                        self.numCoveredEdges[i] += 1 
                    
        r_t = self.getReward(a)  
        self.reward_seq.append(r_t)  
        self.sum_rewards.append(r_t)                 
        return r_t

    def stepWithoutReward(self, a):
     
        assert self.graph
        assert a not in self.covered_set
        self.covered_set.add(a)  
        self.action_list.append(a)  
        
        for i in range(2):
            for neigh in list(self.graph.adj_list[i][a][1]):            
                    if neigh not in self.covered_set and (neigh,a) not in self.remove_edge[i]:
                        self.numCoveredEdges[i] += 1      
        r_t = self.getReward(a)  
        self.score += -1 * r_t
        self.MaxCCList.append(-1 * r_t * self.graph.num_nodes)
    
    def randomAction(self):
       
        assert self.graph
        self.avail_list.clear()  
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                useful1 = any(neigh not in self.covered_set and (i,neigh) not in self.remove_edge[0] for neigh in self.graph.adj_list[0][i][1])
                useful2 = any(neigh not in self.covered_set and (i,neigh) not in self.remove_edge[1] for neigh in self.graph.adj_list[1][i][1])
                if useful1 and useful2:
                    self.avail_list.append(i)
        assert self.avail_list
        idx = random.choice(self.avail_list)  
        return idx

    def betweenAction(self):
        
        assert self.graph
        adj_dic_origin = {}
        adj_list_reID = []

        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        if i in adj_dic_origin:
                            adj_dic_origin[i].append(neigh)
                        else:
                            adj_dic_origin[i] = [neigh]

        id2node = {num: node for num, node in enumerate(adj_dic_origin)}
        node2id = {node: num for num, node in id2node.items()}
        adj_list_reID = [[node2id[neigh] for neigh in adj_dic_origin[node]] for node in adj_dic_origin]

        BC = self.betweenness(adj_list_reID)  
        maxID = max(range(len(BC)), key=BC.__getitem__)
        idx = id2node[maxID]  

        return idx

    def isTerminal(self):
       
        assert self.graph
        return self.graph.num_edges[0] == (self.numCoveredEdges[0] + len(self.remove_edge[0])/2) or self.graph.num_edges[1] == (self.numCoveredEdges[1] + len(self.remove_edge[1])/2)
        
    def getReward(self,a):
        
        orig_node_num = float(self.graph.num_nodes)
        rank = self.getMaxConnectedNodesNum(a)
        
        # R_LMCC
        r_lmcc = -float(rank) / (self.graph.max_rank * orig_node_num)
        
        # R_Bridge
        # Calculate how many inter-community edges are removed/broken
        # We need to see if 'a' was part of any inter-community edges
        bridge_bonus = 0.0
        
        # Check neighbors of a in G1
        if hasattr(self, 'node2com1'): # Safety check
            deg_cost = 0
            broken_inter_edges = 0
            
            # Layer 1
            if a in self.G1:
                # G1.neighbors might include already covered ones, but step() handles logical removal
                # We care about edges (a, v) that exist currently
                for v in self.G1[a]:
                    if v not in self.covered_set and (a,v) not in self.remove_edge[0]:
                        deg_cost += 1
                        if self.node2com1.get(a) != self.node2com1.get(v):
                            broken_inter_edges += 1
                            
            # Layer 2
            if a in self.G2:
                for v in self.G2[a]:
                    if v not in self.covered_set and (a,v) not in self.remove_edge[1]:
                        deg_cost += 1
                        if self.node2com2.get(a) != self.node2com2.get(v):
                            broken_inter_edges += 1
            
            f_het = self.graph.node_features[a][0] if self.graph.node_features is not None else 0
            
            if f_het > self.tau:
                 bridge_bonus = broken_inter_edges / (deg_cost + 1e-6)
        
        return r_lmcc + self.beta * bridge_bonus

        #return -float(self.getRemainingCNDScore()) / (orig_node_num * orig_node_num * (orig_node_num - 1) / 2)

    def getMaxConnectedNodesNum(self,a=None):
        
        assert self.graph
        if self.flag == 0 :
            self.G1 = nx.Graph()
            self.G2 = nx.Graph()
            self.G1.add_nodes_from(range(0,self.graph.num_nodes))
            self.G2.add_nodes_from(range(0,self.graph.num_nodes))
            for i,neighbors in self.graph.adj_list[0]:
                for j in neighbors:
                    if i not in self.covered_set and j not in self.covered_set and (i,j) not in self.remove_edge[0]:
                        self.G1.add_edge(i,j)
            for i,neighbors in self.graph.adj_list[1]:
                for j in neighbors:
                    if i not in self.covered_set and j not in self.covered_set and (i,j) not in self.remove_edge[1]:
                        self.G2.add_edge(i,j)
            self.flag = 1
        else :
            self.G1.remove_node(a)
            self.G2.remove_node(a)
        connected_components = Mcc.MCC(self.G1,self.G2,self.remove_edge)
        rank = Mcc.find_max_set_length(connected_components)
        return float(rank)

    def betweenness(self, adj_list):
     
        nvertices = len(adj_list)
        CB = [0.0] * nvertices
        norm = float((nvertices - 1) * (nvertices - 2))

        for i in range(nvertices):
            PredList = [[] for _ in range(nvertices)]
            d = [float('inf')] * nvertices
            d[i] = 0
            sigma = [0] * nvertices
            sigma[i] = 1
            delta = [0.0] * nvertices
            Q = [i]
            S = []

            while Q:
                u = Q.pop(0)
                S.append(u)
                for neigh in adj_list[u]:
                    v = neigh
                    if d[v] == float('inf'):
                        d[v] = d[u] + 1
                        Q.append(v)
                    if d[v] == d[u] + 1:
                        sigma[v] += sigma[u]
                        PredList[v].append(u)

            while S:
                u = S.pop()
                for pred in PredList[u]:
                    delta[pred] += (sigma[pred] / sigma[u]) * (1 + delta[u])
                if u != i:
                    CB[u] += delta[u]

            PredList.clear()
            d.clear()
            sigma.clear()
            delta.clear()

        for i in range(nvertices):
            if norm == 0:
                CB[i] = 0
            else:
                CB[i] = CB[i] / norm

        return CB
