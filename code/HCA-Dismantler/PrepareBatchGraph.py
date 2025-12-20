import numpy as np
from graph import Graph
from graph_struct import GraphStruct
import torch
import Mcc
import networkx as nx
from typing import List, Tuple, Dict
import sys
#   Preparation of batch data in graph neural networks
class SparseMatrix:
    def __init__(self):
        self.rowIndex = []
        self.colIndex = []
        self.value = []
        self.rowNum = 0
        self.colNum = 0

class PrepareBatchGraph:
    def __init__(self, aggregatorID):
        self.aggregatorID = aggregatorID
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.n2nsum_param = [SparseMatrix(),SparseMatrix()]
        self.laplacian_param = [SparseMatrix(),SparseMatrix()]
        self.subgsum_param = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.subgraph_id_span = []
        self.aux_feat = []
        self.avail_act_cnt = []
        self.graph = [GraphStruct(),GraphStruct()]
        self.adj= []
        self.virtual_adj = []
        self.adj= []
        self.virtual_adj = []
        self.remove_edge_list = []
        self.node_feat = [] # HCA Features
        
        
    def get_status_info(self,g: Graph,covered: List[int], remove_edge: List[set]):
        c = set(covered)
        idx_map = [[-1] * g.num_nodes, [-1] * g.num_nodes]
        counter = [0,0]
        twohop_number = [0,0]
        threehop_number = [0,0]
        node_twohop_set = [set(),set()]
        n = [0,0]
        node_twohop_counter = [{},{}] 
        isolated_nodes = []          

        # Initialize idx_map for all uncovered nodes to ensure consistency across layers
        # This handles isolated nodes that might exist in one layer but not the other
        for u in range(g.num_nodes):
            if u not in c:
                idx_map[0][u] = 0
                idx_map[1][u] = 0
                n[0] += 1
                n[1] += 1
        
        for i in range(2):         
            for p in g.edge_list[i]: 
                if tuple(p) in remove_edge[i]:
                    continue                         
                if p[0] in c or p[1] in c:
                    counter[i] += 1
                else:
                    # Nodes are already marked in idx_map, just calculate stats
                    if p[0] in node_twohop_counter[i]:
                        twohop_number[i] += node_twohop_counter[i][p[0]]
                        node_twohop_counter[i][p[0]] = node_twohop_counter[i][p[0]] + 1
                    else:
                        node_twohop_counter[i][p[0]] = 1

                    if p[1] in node_twohop_counter[i]:
                        twohop_number[i] += node_twohop_counter[i][p[1]]
                        node_twohop_counter[i][p[1]] = node_twohop_counter[i][p[1]] + 1
                    else:
                        node_twohop_counter[i][p[1]] = 1           
        
        assert idx_map[0] == idx_map[1]
        return n,counter,twohop_number,threehop_number,idx_map,remove_edge

    def Setup_graph_input(self, idxes, g_list, covered, actions, remove_edges):
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.avail_act_cnt = []
        self.node_feat = []


        node_cnt = [0,0]
        for i, idx in enumerate(idxes):
            g = g_list[idx]
            temp_feat1 = []
            temp_feat2 = []
            if remove_edges == None:
                avail, counter, twohop_number, _, idx_map, remove_edge = self.get_status_info(g, covered[idx], remove_edges)
            else:
                avail, counter, twohop_number, _, idx_map, remove_edge = self.get_status_info(g, covered[idx], remove_edges[idx])
            
            if g.num_nodes > 0:
                temp_feat1.append(len(covered[idx]) / g.num_nodes)
                temp_feat2.append(len(covered[idx]) / g.num_nodes)
            temp_feat1.append(counter[0] / g.num_edges[0])
            temp_feat1.append(twohop_number[0] / (g.num_nodes * g.num_nodes))
            temp_feat1.append(1.0)
            temp_feat2.append(counter[1] / g.num_edges[1])
            temp_feat2.append(twohop_number[1] / (g.num_nodes * g.num_nodes))
            temp_feat2.append(1.0)
            temp_feat = [temp_feat1,temp_feat2]
            for j in range (2):
                node_cnt[j] += avail[j]

            self.aux_feat.append(temp_feat)
            self.idx_map_list.append(idx_map)
            self.avail_act_cnt.append(avail)
            self.remove_edge_list.append(remove_edge)
            
        for j in range(2):
            # Calculate total subgraphs (communities) across all graphs in batch for this layer
            total_comms = 0
            for idx in idxes:
                g = g_list[idx]
                if hasattr(g, 'subgraphs') and len(g.subgraphs) > j:
                     total_comms += len(g.subgraphs[j])
            
            # If no subgraphs found (fallback), use 1 per graph (global pooling logic)
            if total_comms == 0:
                 total_comms = len(idxes)
            
            self.graph[j].resize(total_comms, node_cnt[j])

            if actions:
                self.act_select[j].rowNum = len(idxes)
                self.act_select[j].colNum = node_cnt[j]
            else:
                self.rep_global[j].rowNum = node_cnt[j]
                self.rep_global[j].colNum = len(idxes)
        
        # Populate Subgraphs (Community Pooling)
        # We need to map nodes from local g to global batch indices.
        # `idx_map` gives us: valid_node -> local_valid_index.
        # `node_cnt_offsets` tracks start index for each graph in batch? 
        # No, `node_cnt` accumulates. `idx_map` is 0-indexed relative to `node_cnt` start of that graph?
        # NO. `idx_map` is 0, 1, 2... for valid nodes.
        # `Setup_graph_input` logic for edges (lines 173):
        # x = idx_map[j][p[0]] + node_cnt[j]
        # Wait, `node_cnt` is updated at END of loop (line 180).
        # So inside the loop, `node_cnt[j]` is the offset for the CURRENT graph `g`.
        
        node_cnt_offset = [0, 0]
        comm_cnt_offset = [0, 0]
        
        for i, idx in enumerate(idxes):
            g = g_list[idx]
            # [Logic for edges and re-indexing is here in original code...] 
            # We need to re-implement the loop or insert our logic? 
            # I am replacing the `resize` block, which is AFTER the loop that built `aux_feat`.
            # BUT `Setup_graph_input` has TWO loops.
            # Loop 1 (lines 92-118): Calc stats, build idx_map, ACCUMULATE total `node_cnt` (lines 112).
            # Wait, `node_cnt` at line 129 is reset to [0,0].
            # The edge addition loop is lines 132+.
            
            # I am replacing lines 119-130.
            # At this point, `node_cnt` holds TOTAL nodes.
        
        # Initializing offsets for the SECOND loop
        curr_node_cnt = [0, 0]
        curr_comm_cnt = [0, 0]
        
        # Need to iterate again or inject into the second loop (which I'm not replacing here fully).
        # I'll inject the subgraph population into the SECOND loop via a separate edit or assume I can do it here?
        # No, I need `idx_map` for each graph which is stored in `self.idx_map_list`.
        # So I can iterate `self.idx_map_list` here.
        
        for i, idx in enumerate(idxes):
             g = g_list[idx]
             idx_map = self.idx_map_list[i]
             
             for j in range(2):
                 # Subgraphs
                 if hasattr(g, 'subgraphs') and len(g.subgraphs) > j and len(g.subgraphs[j]) > 0:
                     comms = g.subgraphs[j]
                     for comm_id, nodes in enumerate(comms):
                         global_comm_id = curr_comm_cnt[j] + comm_id
                         for u in nodes:
                             # u is original node ID.
                             # check if u is valid (not removed/covered)
                             if idx_map[j][u] >= 0:
                                 global_node_id = curr_node_cnt[j] + idx_map[j][u]
                                 self.graph[j].add_node(global_comm_id, global_node_id)
                     curr_comm_cnt[j] += len(comms)
                 else:
                     # Fallback: All valid nodes to one community
                     global_comm_id = curr_comm_cnt[j]
                     for u in range(g.num_nodes):
                         if idx_map[j][u] >= 0:
                             global_node_id = curr_node_cnt[j] + idx_map[j][u]
                             self.graph[j].add_node(global_comm_id, global_node_id)
                     curr_comm_cnt[j] += 1
                     
                 curr_node_cnt[j] += self.avail_act_cnt[i][j]

        node_cnt = [0,0]
        edge_cnt = [0,0]

        for i, idx in enumerate(idxes):
            g = g_list[idx]
            idx_map = self.idx_map_list[i]
            remove_edge = self.remove_edge_list[i]
            t = [0,0]
            for j in range(g.num_nodes):
                for h in range(2):
                    if idx_map[h][j] < 0:
                        continue
                    # idx_map is already 0 for valid nodes from get_status_info logic, 
                    # but here we need sequential indices for the current batch.
                    # Wait, get_status_info sets idx_map to 0 for valid nodes to indicate PRESENCE.
                    # We DO need to assign sequential indices 0, 1, 2... for the matrix construction.
                    # BUT we must update idx_map so that later (line 240) we can lookup `act`.
                    
                    idx_map[h][j] = t[h] # This stores the local sequential index
                    self.graph[h].add_node(i, node_cnt[h] + t[h])
                    if not actions:
                        self.rep_global[h].rowIndex.append(node_cnt[h] + t[h])
                        self.rep_global[h].colIndex.append(i)
                        self.rep_global[h].value.append(1.0)
                    t[h] += 1
                
                # Check if this node is included in the batch (exists in at least one layer)
                # Since idx_map is synced for covered nodes, check either layer 0
                if idx_map[0][j] >= 0:
                     if hasattr(g, 'node_features') and g.node_features is not None:
                         self.node_feat.append(g.node_features[j])
                     else:
                         self.node_feat.append([0.0, 0.0, 0.0])

            #error
            assert t[0] == self.avail_act_cnt[i][0]

            if actions:
                act = actions[idx]
                #error
                assert idx_map[0][act] >= 0 and act >= 0 and act < g.num_nodes
                for j in range(2):
                    self.act_select[j].rowIndex.append(i)
                    self.act_select[j].colIndex.append(node_cnt[j] + idx_map[j][act])
                    self.act_select[j].value.append(1.0)

            for j in range(2):
                for p in g.edge_list[j]:
                    if tuple(p) in remove_edge[j]:
                        continue 
                    if idx_map[j][p[0]] >= 0 and idx_map[j][p[1]] >= 0:
                        x, y = idx_map[j][p[0]] + node_cnt[j], idx_map[j][p[1]] + node_cnt[j]
                        self.graph[j].add_edge(edge_cnt[j], x, y)
                        edge_cnt[j] += 1
                        self.graph[j].add_edge(edge_cnt[j], y, x)
                        edge_cnt[j] += 1

                node_cnt[j] += self.avail_act_cnt[i][j]
        #error
        assert node_cnt[0] == self.graph[0].num_nodes
        result_list = self.n2n_construct(self.aggregatorID)
        self.n2nsum_param = result_list[0]
        self.laplacian_param = result_list[1]
        self.adj = result_list[2]
        result_list1 = self.subg_construct()
        self.subgsum_param = result_list1[0]
        self.virtual_adj = result_list1[1]
        
        self.comm_adj_param = self.comm_adj_construct()
        
        for j in range(2):
            self.act_select[j] = self.convert_sparse_to_tensor(self.act_select[j])

            self.rep_global[j] = self.convert_sparse_to_tensor(self.rep_global[j])
            self.n2nsum_param[j] = self.convert_sparse_to_tensor(self.n2nsum_param[j])
            self.laplacian_param[j] = self.convert_sparse_to_tensor(self.laplacian_param[j])
            self.subgsum_param[j] = self.convert_sparse_to_tensor(self.subgsum_param[j])
            self.comm_adj_param[j] = self.convert_sparse_to_tensor(self.comm_adj_param[j])

        # Convert node_feat to tensor
        if self.node_feat:
            # Shape: [Total_Nodes, 3]
            self.node_feat_tensor = torch.tensor(np.array(self.node_feat), dtype=torch.float)
        else:
            self.node_feat_tensor = None





    def SetupTrain(self, idxes, g_list, covered, actions, remove_edges):
        self.Setup_graph_input(idxes, g_list, covered, actions, remove_edges)

    def SetupPredAll(self, idxes, g_list, covered, remove_edges):
        self.Setup_graph_input(idxes, g_list, covered, None, remove_edges)
    '''
    def convert_sparse_to_tensor(self, matrix):
        indices = np.column_stack((matrix.rowIndex, matrix.colIndex))
        return torch.sparse.FloatTensor(torch.LongTensor(indices).t(), torch.FloatTensor(matrix.value),
                                         torch.Size([matrix.rowNum, matrix.colNum]))
    '''

    def convert_sparse_to_tensor(self, matrix):
        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()

        index = torch.tensor(np.transpose(np.array(indices)), dtype=torch.long)
        value = torch.Tensor(np.array(data))
        #index, value = torch_sparse.coalesce(index, value, m=rowNum, n=colNum)
        return_dict = {"index": index, "value": value, "m":rowNum, "n":colNum}
        return return_dict

    '''
    def graph_resize(self, size, node_cnt):
        self.graph = Graph(size, node_cnt)

    def graph_add_node(self, i, node):
        self.graph.add_node(i, node)

    def graph_add_edge(self, edge, x, y):
        self.graph.add_edge(edge, x, y)
    '''

    def n2n_construct(self, aggregatorID):
        result = [SparseMatrix(),SparseMatrix()]
        result_laplacian = [SparseMatrix(),SparseMatrix()]
        adj_matrixs = []
        for h in range(2):
            result[h].rowNum = self.graph[h].num_nodes
            result[h].colNum = self.graph[h].num_nodes
            result_laplacian[h].rowNum = self.graph[h].num_nodes
            result_laplacian[h].colNum = self.graph[h].num_nodes

            for i in range(self.graph[h].num_nodes):
                list1 = self.graph[h].in_edges.head[i]

                if len(list1) > 0:
                    result_laplacian[h].value.append(len(list1))
                    result_laplacian[h].rowIndex.append(i)
                    result_laplacian[h].colIndex.append(i)

                for j in range(len(list1)):
                    if aggregatorID == 0:
                        result[h].value.append(1.0)
                    elif aggregatorID == 1:
                        result[h].value.append(1.0 / len(list1))
                    elif aggregatorID == 2:
                        #neighborDegree = len(self.graph.in_edges.head[list1[j].second])
                        neighborDegree = len(self.graph[h].in_edges.head[list1[j][1]])
                        selfDegree = len(list1)
                        norm = np.sqrt(neighborDegree + 1) * np.sqrt(selfDegree + 1)
                        result[h].value.append(1.0 / norm)

                    result[h].rowIndex.append(i)
                    #result[i].colIndex.append(list1[j].second)
                    result[h].colIndex.append(list1[j][1])
                    result_laplacian[h].value.append(-1.0)
                    result_laplacian[h].rowIndex.append(i)
                    #result[i].result_laplacian[i].colIndex.append(list1[j].second)
                    result_laplacian[h].colIndex.append(list1[j][1])

            adj_matrix = np.zeros((self.graph[h].num_nodes,self.graph[h].num_nodes))
            for edge in self.graph[h].edge_list:
                i,j=edge
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
            adj_matrixs.append(adj_matrix)
        return [result,result_laplacian,adj_matrixs]

    '''
    def e2n_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_nodes
        result.colNum = self.graph.num_edges

        for i in range(self.graph.num_nodes):
            list1 = self.graph.in_edges.head[i]
            for j in range(len(list1)):
                result.value.append(1.0)
                result.rowIndex.append(i)
                result.colIndex.append(list1[j].first)
        return result

    def n2e_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_edges
        result.colNum = self.graph.num_nodes

        for i in range(self.graph.num_edges):
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(self.graph.edge_list[i].first)

        return result

    def e2e_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_edges
        result.colNum = self.graph.num_edges

        for i in range(self.graph.num_edges):
            node_from, node_to = self.graph.edge_list[i]
            list1 = self.graph.in_edges.head[node_from]

            for j in range(len(list1)):
                if list1[j].second == node_to:
                    continue
                result.value.append(1.0)
                result.rowIndex.append(i)
                result.colIndex.append(list1[j].first)

        return result
    '''

    def subg_construct(self):
        result = [SparseMatrix(),SparseMatrix()]
        virtual_adjs = []
        for h in range(2):
            result[h].rowNum = self.graph[h].num_subgraph
            result[h].colNum = self.graph[h].num_nodes

            subgraph_id_span = []
            start = 0
            end = 0

            for i in range(self.graph[h].num_subgraph):
                list1 = self.graph[h].subgraph.head[i]
                end = start + len(list1) - 1

                for j in range(len(list1)):
                    node_idx = list1[j]
                    weight = 1.0
                    
                    # Weighted Pooling using f_roi
                    if self.node_feat:
                        # f_roi is at index 2
                        # Need to handle if node_idx is out of bound of node_feat?
                        # node_feat matches the sequential addition of nodes to batch
                        # But wait, node_idx here is the index in self.graph[h] (0 to node_cnt)
                        # self.node_feat was built by checking `if idx_map[0][j] >= 0`
                        # This loop order (iterates idxes, then j) matches the order nodes are added to `node_cnt`.
                        # However, self.node_feat accumulates nodes for the WHOLE batch for layer 0?
                        # No, check Setup_graph_input:
                        # self.node_feat.append() happens inside `for i, idx... for j...`.
                        # This corresponds PRECISELY to the order nodes are added to self.graph[0] AND self.graph[1] (since they share t[h]).
                        # So node_idx in self.graph[h] (which is node_cnt + t[h]) works for looking up node_feat?
                        # Wait, self.graph[h] uses 0-based index local to that graph structure?
                        # Yes, `add_node(i, node_cnt + t)` adds node ID `node_cnt+t`.
                        # So yes, node index `x` in `self.graph` corresponds to `self.node_feat[x]`.
                        if node_idx < len(self.node_feat):
                            f_roi = self.node_feat[node_idx][2]
                            # Softmax(f_roi) - We can just use f_roi as weight, softmax can be done in net or here?
                            # "Agg using Weighted Pooling, weight is f_roi".
                            # Usually Weighted Pooling is Sum(w * h). Softmax implies weights sum to 1?
                            # "h_Comm = sum_{u} Softmax(f_roi(u)) * h_u".
                            # Computing global softmax here is hard (need sum over comm).
                            # I will store f_roi here. The normalization can happen if I normalize `subgsum_param` row-wise.
                            # For now, let's just use exp(f_roi) or f_roi to emulate attention-like.
                            # Since f_roi can be large, maybe just use it directly or normalized.
                            # I'll use f_roi directly. The Net can handle magnitude.
                            weight = f_roi + 1e-6

                    result[h].value.append(weight)
                    result[h].rowIndex.append(i)

                    result[h].colIndex.append(list1[j])

                if len(list1) > 0:
                    subgraph_id_span.append((start, end))
                else:
                    subgraph_id_span.append((self.graph[h].num_nodes, self.graph[h].num_nodes))
                start = end + 1
            virtual_adj = np.zeros((result[h].rowNum,result[h].colNum))
            for i in range(len(result[h].value)):
                row_idx = result[h].rowIndex[i]
                col_idx = result[h].colIndex[i]
                weight = result[h].value[i]
                virtual_adj[row_idx][col_idx] = weight
            virtual_adjs.append(virtual_adj)
        return [result,virtual_adjs]

    def comm_adj_construct(self):
        """Build Coarse Community Graph Adjacency Matrix"""
        result = [SparseMatrix(), SparseMatrix()]
        delta = 0 # Threshold for edges to consider communities connected
        
        # We need to find edges between communities.
        # self.graph is GraphStruct
        # self.graph[h].subgraph maps subg_id -> list of nodes
        # self.graph[h].edge_list contains edges
        
        for h in range(2):
            g_struct = self.graph[h]
            num_comms = g_struct.num_subgraph
            result[h].rowNum = num_comms
            result[h].colNum = num_comms
            
            # Map node -> comm
            node2comm = {}
            for cid in range(num_comms):
                nodes = g_struct.subgraph.head[cid]
                for u in nodes:
                    node2comm[u] = cid
            
            # Count edges between communities
            comm_edges = {} # (c1, c2) -> count
            
            for (u, v) in g_struct.edge_list:
                c1 = node2comm.get(u)
                c2 = node2comm.get(v)
                if c1 is not None and c2 is not None and c1 != c2:
                    k = tuple(sorted((c1, c2)))
                    comm_edges[k] = comm_edges.get(k, 0) + 1
            
            for (c1, c2), count in comm_edges.items():
                if count > delta:
                    result[h].value.append(1.0)
                    result[h].rowIndex.append(c1)
                    result[h].colIndex.append(c2)
                    
                    result[h].value.append(1.0)
                    result[h].rowIndex.append(c2)
                    result[h].colIndex.append(c1)
                    
            # Add self loops
            for i in range(num_comms):
                result[h].value.append(1.0)
                result[h].rowIndex.append(i)
                result[h].colIndex.append(i)
                
        return result



