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
        self.remove_edge_list = []
        self.community_features = []
        self.community_partitions = []
        self.comm_pool = {"index": [], "value": [], "m": 0, "n": 0}
        self.node_feat = []
        self.community_index = []
        self.community_batch = []
        
    def get_status_info(self,g: Graph,covered: List[int], remove_edge: List[set]):
        # ensure remove_edge is always a 2-element list of sets
        if remove_edge is None:
            remove_edge = [set(), set()]
        c = set(covered)
        idx_map = [[-1] * g.num_nodes, [-1] * g.num_nodes]
        counter = [0,0]
        twohop_number = [0,0]
        threehop_number = [0,0]
        node_twohop_set = [set(),set()]
        n = [0,0]
        node_twohop_counter = [{},{}] 
        isolated_nodes = []          
        for i in range(2):         
            for p in g.edge_list[i]: 
                if tuple(p) in remove_edge[i]:
                    continue                         
                ##if p[0] in c_new_set or p[1] in c_new_set:
                if p[0] in c or p[1] in c:
                    counter[i] += 1
                else:
                    if idx_map[i][p[0]] < 0:
                        n[i] += 1

                    if idx_map[i][p[1]] < 0:
                        n[i] += 1

                    idx_map[i][p[0]] = 0
                    idx_map[i][p[1]] = 0

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
        # ensure both layers share the same node index map; some graphs may differ per layer
        if idx_map[0] != idx_map[1]:
            idx_map[1] = idx_map[0][:]
        return n,counter,twohop_number,threehop_number,idx_map,remove_edge

    def Setup_graph_input(self, idxes, g_list, covered, actions, remove_edges):
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.avail_act_cnt = []
        self.community_features = []
        self.community_partitions = []
        self.comm_pool = {"index": [], "value": [], "m": 0, "n": 0}
        self.node_feat = [[], []]
        self.community_index = []
        self.community_batch = []
        self.node_batch = [[], []]
        self.nodes_list = []

        node_cnt = [0,0]
        total_comm = 0
        for i, idx in enumerate(idxes):
            g = g_list[idx]
            temp_feat1 = []
            temp_feat2 = []
            if remove_edges is None:
                avail, counter, twohop_number, _, idx_map, remove_edge = self.get_status_info(g, covered[i], None)
            else:
                avail, counter, twohop_number, _, idx_map, remove_edge = self.get_status_info(g, covered[i], remove_edges[i])
            
            if g.num_nodes > 0:
                temp_feat1.append(len(covered[i]) / g.num_nodes)
                temp_feat2.append(len(covered[i]) / g.num_nodes)
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
            # capture community stats for downstream consumption
            self.community_features.append({
                "participation": getattr(g, "node_participation", None),
                "zscore": getattr(g, "node_zscore", None)
            })
            self.community_partitions.append(getattr(g, "community_partition", {}))
            
        for j in range(2):
            self.graph[j].resize(len(idxes), node_cnt[j])

            if actions:
                self.act_select[j].rowNum = len(idxes)
                self.act_select[j].colNum = node_cnt[j]
            else:
                self.rep_global[j].rowNum = node_cnt[j]
                self.rep_global[j].colNum = len(idxes)

        node_cnt = [0,0]
        edge_cnt = [0,0]

        for i, idx in enumerate(idxes):
            g = g_list[idx]
            idx_map = self.idx_map_list[i]
            remove_edge = self.remove_edge_list[i]
            comm_part = self.community_partitions[i] if self.community_partitions[i] else {}
            comm_id_map = {}
            t = [0,0]
            # precompute max degree per layer
            max_deg = []
            for h in range(2):
                if len(g.adj_list[h]) == 0:
                    max_deg.append(1)
                else:
                    max_deg.append(max(len(item[1]) for item in g.adj_list[h]))
            for j in range(g.num_nodes):
                for h in range(2):
                    if idx_map[h][j] < 0:
                        continue
                    idx_map[h][j] = t[h]
                    self.graph[h].add_node(i, node_cnt[h] + t[h])
                    if not actions:
                        self.rep_global[h].rowIndex.append(node_cnt[h] + t[h])
                        self.rep_global[h].colIndex.append(i)
                        self.rep_global[h].value.append(1.0)
                    # build node features (deg_norm + community 3 dims)
                    part = 0.0
                    zval = 0.0
                    if self.community_features[i]["participation"] is not None:
                        part = float(self.community_features[i]["participation"][j])
                    if self.community_features[i]["zscore"] is not None:
                        zval = float(self.community_features[i]["zscore"][j])
                    comm_feat = [part, zval]
                    deg = len(g.adj_list[h][j][1]) if len(g.adj_list[h]) > j else 0
                    deg_norm = deg / max_deg[h] if max_deg[h] > 0 else 0.0
                    feat_vec = [deg_norm] + list(comm_feat)
                    if len(self.node_feat[h]) <= node_cnt[h] + t[h]:
                        self.node_feat[h].append(feat_vec)
                    else:
                        self.node_feat[h][node_cnt[h] + t[h]] = feat_vec
                    # record node -> graph mapping for batch-wise scatter
                    self.node_batch[h].append(i)
                    # record community id mapping for comm_pool (use layer0 mapping)
                    if h == 0:
                        cid = comm_part.get(j, 0)
                        comm_id_map[j] = cid
                    t[h] += 1
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
                        # defensive: skip any edge that would exceed allocated node count
                        if x >= self.graph[j].num_nodes or y >= self.graph[j].num_nodes:
                            continue
                        self.graph[j].add_edge(edge_cnt[j], x, y)
                        edge_cnt[j] += 1
                        self.graph[j].add_edge(edge_cnt[j], y, x)
                        edge_cnt[j] += 1

                node_cnt[j] += self.avail_act_cnt[i][j]
            # build comm_pool indices for this graph using layer0 mapping with contiguous community ids
            if comm_part:
                raw_ids = sorted(set(comm_part.values()))
                local_comm_map = {cid: k for k, cid in enumerate(raw_ids)}
                current_comm_count = len(raw_ids)
                offset_comm = total_comm
                total_comm += current_comm_count
                for orig_node, cid in comm_part.items():
                    mapped = idx_map[0][orig_node]
                    if mapped >= 0:
                        mapped_cid = local_comm_map[cid]
                        final_cid = offset_comm + mapped_cid
                        self.comm_pool["index"].append([final_cid, node_cnt[0] - self.avail_act_cnt[i][0] + mapped])
                        self.comm_pool["value"].append(1.0)
                        self.community_index.append(final_cid)
                # one community_batch entry per community
                for _ in range(current_comm_count):
                    self.community_batch.append(i)
            else:
                # single community fallback
                offset_comm = total_comm
                total_comm += 1
                for orig_node in range(g.num_nodes):
                    mapped = idx_map[0][orig_node]
                    if mapped >= 0:
                        self.comm_pool["index"].append([offset_comm, node_cnt[0] - self.avail_act_cnt[i][0] + mapped])
                        self.comm_pool["value"].append(1.0)
                        self.community_index.append(offset_comm)
                self.community_batch.append(i)
        #error
        assert node_cnt[0] == self.graph[0].num_nodes
        self.comm_pool["m"] = total_comm
        self.comm_pool["n"] = node_cnt[0]
        # ensure node_feat and node_batch lengths are aligned per layer
        for h in range(2):
            min_len = min(len(self.node_feat[h]), len(self.node_batch[h]))
            if len(self.node_feat[h]) != len(self.node_batch[h]):
                self.node_feat[h] = self.node_feat[h][:min_len]
                self.node_batch[h] = self.node_batch[h][:min_len]
        # build nodes list aligned to active nodes (layer 0)
        self.nodes_list = list(range(len(self.node_feat[0])))
        if self.comm_pool["index"]:
            indices = np.array(self.comm_pool["index"]).T
            values = np.array(self.comm_pool["value"])
            self.comm_pool = {
                "index": torch.tensor(indices, dtype=torch.long),
                "value": torch.tensor(values, dtype=torch.float),
                "m": total_comm,
                "n": node_cnt[0],
            }
        else:
            self.comm_pool = {
                "index": torch.zeros((2,0), dtype=torch.long),
                "value": torch.zeros((0,), dtype=torch.float),
                "m": 0,
                "n": node_cnt[0],
            }
        # convert node_feat to numpy arrays
        self.node_feat = [np.array(f, dtype=np.float32) if len(f)>0 else np.zeros((0,3),dtype=np.float32) for f in self.node_feat]
        # node_batch to tensors
        self.node_batch = [torch.tensor(b, dtype=torch.long) for b in self.node_batch]
        # nodes list tensor
        self.nodes_list = torch.tensor(self.nodes_list, dtype=torch.long)
        # Final alignment check
        if not (len(self.node_feat[0]) == len(self.node_batch[0]) == len(self.nodes_list)):
            raise RuntimeError(f"Data Alignment Error! Feat: {len(self.node_feat[0])}, Batch: {len(self.node_batch[0])}, Nodes: {len(self.nodes_list)}")
        # convert community index/batch
        self.community_index = torch.tensor(self.community_index, dtype=torch.long)
        self.community_batch = torch.tensor(self.community_batch, dtype=torch.long)
        result_list = self.n2n_construct(self.aggregatorID)
        self.n2nsum_param = result_list[0]
        self.laplacian_param = result_list[1]
        self.adj = result_list[2]
        result_list1 = self.subg_construct()
        self.subgsum_param = result_list1[0]
        self.virtual_adj = result_list1[1]
        for j in range(2):
            self.act_select[j] = self.convert_sparse_to_tensor(self.act_select[j])
            self.rep_global[j] = self.convert_sparse_to_tensor(self.rep_global[j])
            self.n2nsum_param[j] = self.convert_sparse_to_tensor(self.n2nsum_param[j])
            self.laplacian_param[j] = self.convert_sparse_to_tensor(self.laplacian_param[j])
            self.subgsum_param[j] = self.convert_sparse_to_tensor(self.subgsum_param[j])
        # keep comm_pool as tensors on CPU; move later in torch code

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
        # enforce consistent shapes across layers to avoid ragged adjacency
        max_rows = max(self.graph[0].num_subgraph, self.graph[1].num_subgraph)
        max_cols = max(self.graph[0].num_nodes, self.graph[1].num_nodes)
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
                    result[h].value.append(1.0)
                    result[h].rowIndex.append(i)
                    result[h].colIndex.append(list1[j])

                if len(list1) > 0:
                    subgraph_id_span.append((start, end))
                else:
                    subgraph_id_span.append((self.graph[h].num_nodes, self.graph[h].num_nodes))
                start = end + 1
            # build dense adjacency with unified shape (max_rows x max_cols)
            virtual_adj = np.zeros((max_rows, max_cols))
            for i in range(len(result[h].value)):
                row_idx = result[h].rowIndex[i]
                col_idx = result[h].colIndex[i]
                weight = result[h].value[i]
                if row_idx < 0 or col_idx < 0:
                    continue
                if row_idx >= max_rows or col_idx >= max_cols:
                    continue
                virtual_adj[row_idx][col_idx] = weight
            virtual_adjs.append(virtual_adj)
        return [result,virtual_adjs]
