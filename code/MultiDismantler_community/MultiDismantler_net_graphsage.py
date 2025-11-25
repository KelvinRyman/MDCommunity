from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
from torch_scatter import scatter_mean
import numpy as np
from MRGNN.encoders import Encoder
from MRGNN.aggregators import MeanAggregator, LSTMAggregator, PoolAggregator
from MRGNN.utils import LogisticRegression
from MRGNN.mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention, \
    BitwiseMultipyLogis
import sys
# cudnn.benchmark = False
# cudnn.deterministic = True
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# out = torch_sparse.spmm(index, value, m, n, matrix)
class MultiDismantler_net(nn.Module):
    def __init__(self, layerNodeAttention_weight,
                 embedding_size=64, w_initialization_std=1, reg_hidden=32, max_bp_iter=3,
                 embeddingMethod=1, aux_dim=4, device=None, node_attr=False, node_feat_dim=3):
        super(MultiDismantler_net, self).__init__()

        self.layerNodeAttention_weight = layerNodeAttention_weight
        # self.rand_generator = torch.normal
        # see https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
        self.rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size), 2)
        self.embedding_size = embedding_size
        self.w_initialization_std = w_initialization_std
        self.reg_hidden = reg_hidden
        self.max_bp_iter = max_bp_iter
        self.embeddingMethod = embeddingMethod
        self.aux_dim = aux_dim
        self.device = device
        self.node_attr = node_attr
        self.node_feat_dim = node_feat_dim
        self.act = nn.ReLU()
        self.node_encoder = nn.Linear(self.node_feat_dim, self.embedding_size)
        self.comm_transformer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=4, batch_first=True)
        
        # [2, embed_dim]
        self.w_n2l = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(2, self.embedding_size)))
        # [embed_dim, embed_dim]
        self.p_node_conv = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.embedding_size, self.embedding_size)))
        
        self.p_node_conv2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                                size=(self.embedding_size,
                                                                                      self.embedding_size)))
            # [2*embed_dim, embed_dim]
        self.p_node_conv3 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                                size=(2 * self.embedding_size,
                                                                                      self.embedding_size)))

        # [reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(
                                                                                 self.embedding_size, self.reg_hidden)))

            # [reg_hidden+aux_dim, 1]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            self.h2_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.reg_hidden + self.aux_dim + 2*self.embedding_size, 1)))
            # [reg_hidden2 + aux_dim, 1]
            self.last_w = self.h2_weight
        else:
            # [2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(
                                                                                 2 * self.embedding_size,
                                                                                 self.reg_hidden)))
            # [2*embed_dim, reg_hidden]
            self.last_w = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(2 * self.embedding_size + self.aux_dim + 2*self.embedding_size, 1)))

        ## [embed_dim, 1]
        # cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)
        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.embedding_size, 1)))
        #self.w_layer = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     #size=(embedding_size, 1)))
        self.w_layer1 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(embedding_size, 128)))
        self.w_layer2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(128, 1)))

        self.comm_attention = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1)
        )
        self.debug_counter = 0
        self.debug_comm = False

        # Dueling streams: value uses global state, advantage uses global + node
        value_hidden = self.reg_hidden if self.reg_hidden > 0 else self.embedding_size
        adv_hidden = self.reg_hidden if self.reg_hidden > 0 else self.embedding_size
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_size, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1)
        )
        self.adv_mlp = nn.Sequential(
            nn.Linear(3 * self.embedding_size, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, 1)
        )
        
        self.flag = 0
    def train_forward(self, node_input, subgsum_param, n2nsum_param, action_select, aux_input, adj, v_adj, comm_pool=None, community_index=None, community_batch=None, node_batch=None):
        nodes_cnt = n2nsum_param[0]['m']
        use_external_feat = node_input is not None
        if use_external_feat:
            if isinstance(node_input, list):
                node_input = [ni.to(self.device) for ni in node_input]
            else:
                node_input = node_input.to(self.device)
        else:
            node_input = torch.zeros((2, nodes_cnt, 2)).to(self.device)                       
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((2, y_nodes_size, 2)).to(self.device)
        if isinstance(adj, list):
            adj = [torch.tensor(a, dtype=torch.float, device=self.device) for a in adj]
        else:
            adj = torch.tensor(adj, dtype=torch.float, device=self.device)
        if isinstance(v_adj, list):
            v_adj = [torch.tensor(a, dtype=torch.float, device=self.device) for a in v_adj]
        else:
            v_adj = torch.tensor(v_adj, dtype=torch.float, device=self.device)
        node_embedding = []
        lay_num = 2
        for l in range(lay_num):
            if use_external_feat:
                input_potential_layer = self.node_encoder(node_input[l])
            else:
                for i in range(y_nodes_size):
                    node_in_graph = torch.where(v_adj[l][i] == 1)
                    if node_in_graph[0].numel() == 0:
                        continue
                    degree = torch.sum(adj[l][node_in_graph], axis=1, keepdims=True)
                    degree_max,_ = torch.max(degree,dim=0)
                    degree_new = degree/degree_max
                    node_feature = torch.cat((degree_new,degree_new),axis = 1)
                    node_input[l][node_in_graph] = node_feature
                input_message = torch.matmul(node_input[l], self.w_n2l)
                input_potential_layer = self.act(input_message)
        for l in range(lay_num):
            if use_external_feat:
                # after encoder already in embedding space
                input_potential_layer = torch.nn.functional.normalize(input_potential_layer, p=2, dim=1)

            # # no sparse
            # [batch_size, embed_dim]
            #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
            y_input_message = torch.matmul(y_node_input[l], self.w_n2l)
            #[batch_size, embed_dim]  # no sparse
            #y_input_potential_layer = tf.nn.relu(y_input_message)
            y_input_potential_layer = self.act(y_input_message)

            lv = 0
            #[node_cnt, embed_dim], no sparse
            cur_message_layer = input_potential_layer
            #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

            #[batch_size, embed_dim], no sparse
            y_cur_message_layer = y_input_potential_layer
            # [batch_size, embed_dim]
            #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            while lv < self.max_bp_iter:
                lv =lv + 1
                node_count = cur_message_layer.size(0)
                n2npool = torch_sparse.spmm(n2nsum_param[l]['index'], n2nsum_param[l]['value'],\
                        node_count, node_count, cur_message_layer)
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                #node_linear = tf.matmul(n2npool, p_node_conv)
                node_linear = torch.matmul(n2npool, self.p_node_conv)
                
                #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
                y_n2npool = torch_sparse.spmm(subgsum_param[l]['index'], subgsum_param[l]['value'],\
                        subgsum_param[l]['m'], node_count, cur_message_layer)

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_node_linear = torch.matmul(y_n2npool, self.p_node_conv)
                
                cur_message_layer_linear = torch.matmul(cur_message_layer, self.p_node_conv2)
                    #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                    #merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                #cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))
                cur_message_layer = self.act(torch.matmul(merged_linear, self.p_node_conv3))
                cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                #y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                y_cur_message_layer_linear = torch.matmul(y_cur_message_layer, self.p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                #y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                #y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))
                y_cur_message_layer = self.act(torch.matmul(y_merged_linear, self.p_node_conv3))
                y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            node_output = torch.cat((cur_message_layer,y_cur_message_layer),axis = 0)
            #node_output = torch.nn.functional.normalize(node_output, p=2, dim=1)
            node_embedding.append(node_output)    
                    
        node_embedding_0 = node_embedding[0]
        node_embedding_1 = node_embedding[1]
        if self.embeddingMethod == 1:  # MRGNN
            embeds = [node_embedding_0,node_embedding_1]
            # use CPU numpy index to stay compatible with downstream NumPy expectations
            nodes = np.arange(node_embedding_0.size(0))
            message_layer = torch.zeros(lay_num, len(nodes), self.embedding_size).to(self.device)
            for l in range(lay_num):
                result_temp = self.layerNodeAttention_weight(embeds, nodes, l)
                message_layer[l] = result_temp
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)
        # message_layer = torch.stack(node_embedding)          
        # cur_message_layer = message_layer[:, :nodes_cnt, :]
        # y_cur_message_layer = message_layer[:, nodes_cnt:, :]
        # hierarchical: node -> community -> global (per-graph)
        S_comm_global = None
        if community_index is not None and len(community_index) > 0 and community_batch is not None and len(community_batch) > 0:
            comm_emb = scatter_mean(cur_message_layer[0], community_index, dim=0)
            S_comm_global = scatter_mean(comm_emb, community_batch, dim=0)
        if S_comm_global is None:
            # fallback: per-graph mean from nodes
            if node_batch is not None and len(node_batch) > 0:
                num_graphs = int(torch.max(node_batch[0]).item()) + 1 if node_batch[0].numel() > 0 else 1
                S_comm_global = scatter_mean(cur_message_layer[0], node_batch[0], dim=0)
                if S_comm_global.size(0) < num_graphs:
                    pad = torch.zeros((num_graphs - S_comm_global.size(0), self.embedding_size), device=self.device)
                    S_comm_global = torch.cat([S_comm_global, pad], dim=0)
            else:
                S_comm_global = torch.mean(cur_message_layer[0], dim=0, keepdim=True)
        # debug monitor every 100 calls (silenced by default)
        if self.debug_comm:
            self.debug_counter += 1
            if self.debug_counter % 100 == 0:
                with torch.no_grad():
                    s_virtual_norm = cur_message_layer.mean(dim=1).norm(dim=1).mean().item()
                    s_comm_norm = S_comm_global.norm().item()
                    alpha_min = alpha.min().item() if 'alpha' in locals() else 0.0
                    alpha_max = alpha.max().item() if 'alpha' in locals() else 0.0
                    print(f"[comm-debug] step {self.debug_counter}: "
                          f"S_virtual_norm={s_virtual_norm:.4f}, "
                          f"S_comm_norm={s_comm_norm:.4f}, "
                          f"alpha_min={alpha_min:.4f}, alpha_max={alpha_max:.4f}")
        q_list = []
        w_layer = []
        for l in range(lay_num):
            if node_batch is not None and len(node_batch) > l and node_batch[l].numel() > 0:
                nb = node_batch[l]
                num_graphs = int(torch.max(nb).item()) + 1
                S_virtual = scatter_mean(cur_message_layer[l], nb, dim=0)
                if S_comm_global.size(0) < num_graphs:
                    pad = torch.zeros((num_graphs - S_comm_global.size(0), self.embedding_size), device=self.device)
                    S_comm_global_graph = torch.cat([S_comm_global, pad], dim=0)
                else:
                    S_comm_global_graph = S_comm_global[:num_graphs]
                S_global = torch.concat([S_virtual, S_comm_global_graph], dim=1)  # [num_graphs, 2*embed]
                S_global_nodes = S_global[nb]
                adv_all = self.adv_mlp(torch.concat([S_global_nodes, cur_message_layer[l]], dim=1))  # [nodes,1]
                mean_adv_graph = scatter_mean(adv_all, nb, dim=0)
                q_all = self.value_mlp(S_global)[nb] + (adv_all - mean_adv_graph[nb])
            else:
                # fallback single-graph
                S_virtual = torch.mean(cur_message_layer[l], dim=0, keepdim=True)
                S_global = torch.concat([S_virtual, S_comm_global if S_comm_global.dim() == 2 else S_comm_global.unsqueeze(0)], dim=1)
                S_global_expand_nodes = S_global.repeat(cur_message_layer[l].size(0), 1)
                adv_all = self.adv_mlp(torch.concat([S_global_expand_nodes, cur_message_layer[l]], dim=1))
                mean_adv_graph = torch.mean(adv_all, dim=0, keepdim=True)
                q_all = self.value_mlp(S_global) + (adv_all - mean_adv_graph)

            q_selected = torch_sparse.spmm(action_select[l]['index'], action_select[l]['value'],
                                           action_select[l]['m'], action_select[l]['n'], q_all)

            y_potential = y_cur_message_layer[l]
            w_layer.append((self.act(y_potential @ self.w_layer1))@self.w_layer2)
            q_list.append(q_selected)
        w_layer = torch.concat(w_layer,dim = 1)
        w_layer_softmax = F.softmax(w_layer,dim = 1)
        q = w_layer_softmax[:,0].unsqueeze(1) * q_list[0] + w_layer_softmax[:,1].unsqueeze(1) * q_list[1]
        return q, cur_message_layer

    def test_forward(self, node_input, subgsum_param, n2nsum_param, rep_global, aux_input, adj, v_adj, comm_pool=None, community_index=None, community_batch=None, node_batch=None):
        nodes_cnt = n2nsum_param[0]['m']
        use_external_feat = node_input is not None
        if use_external_feat:
            if isinstance(node_input, list):
                node_input = [ni.to(self.device) for ni in node_input]
            else:
                node_input = node_input.to(self.device)
        else:
            node_input = torch.zeros((2, nodes_cnt, 2), dtype=torch.float).to(self.device)                            
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((2, y_nodes_size, 2), dtype=torch.float).to(self.device)
        if isinstance(adj, list):
            adj = [torch.tensor(a, dtype=torch.float, device=self.device) for a in adj]
        else:
            adj = torch.tensor(adj, dtype=torch.float, device=self.device)
        if isinstance(v_adj, list):
            v_adj = [torch.tensor(a, dtype=torch.float, device=self.device) for a in v_adj]
        else:
            v_adj = torch.tensor(v_adj, dtype=torch.float, device=self.device)
        node_embedding = []
        lay_num = 2
        for l in range(lay_num):
            if use_external_feat:
                input_potential_layer = self.node_encoder(node_input[l])
                input_potential_layer = torch.nn.functional.normalize(input_potential_layer, p=2, dim=1)
            else:
                for i in range(y_nodes_size):
                    node_in_graph = torch.where(v_adj[l][i] == 1)
                    if node_in_graph[0].numel() == 0:
                        continue
                    degree = torch.sum(adj[l][node_in_graph], axis=1, keepdims=True)
                    degree_max,_ = torch.max(degree,dim=0)
                    degree_new = degree/degree_max
                    node_feature = torch.cat((degree_new,degree_new),axis = 1)
                    node_input[l][node_in_graph] = node_feature
                input_message = torch.matmul(node_input[l], self.w_n2l)
                #[node_cnt, embed_dim]  # no sparse
                #input_potential_layer = tf.nn.relu(input_message)
                input_potential_layer = self.act(input_message)

            # # no sparse
            # [batch_size, embed_dim]
            #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
            y_input_message = torch.matmul(y_node_input[l], self.w_n2l)
            #[batch_size, embed_dim]  # no sparse
            #y_input_potential_layer = tf.nn.relu(y_input_message)
            y_input_potential_layer = self.act(y_input_message)

            lv = 0
            #[node_cnt, embed_dim], no sparse
            cur_message_layer = input_potential_layer
            #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

            #[batch_size, embed_dim], no sparse
            y_cur_message_layer = y_input_potential_layer
            # [batch_size, embed_dim]
            #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            while lv < self.max_bp_iter:
                lv =lv + 1
                node_count = cur_message_layer.size(0)
                n2npool = torch_sparse.spmm(n2nsum_param[l]['index'], n2nsum_param[l]['value'],\
                        node_count, node_count, cur_message_layer)
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                #node_linear = tf.matmul(n2npool, p_node_conv)
                node_linear = torch.matmul(n2npool, self.p_node_conv)
                
                #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
                y_n2npool = torch_sparse.spmm(subgsum_param[l]['index'], subgsum_param[l]['value'],\
                        subgsum_param[l]['m'], node_count, cur_message_layer)

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_node_linear = torch.matmul(y_n2npool, self.p_node_conv)
                
                cur_message_layer_linear = torch.matmul(cur_message_layer, self.p_node_conv2)
                    #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                    #merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                #cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))
                cur_message_layer = self.act(torch.matmul(merged_linear, self.p_node_conv3))
                cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
                
                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                #y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                y_cur_message_layer_linear = torch.matmul(y_cur_message_layer, self.p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                #y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                #y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))
                y_cur_message_layer = self.act(torch.matmul(y_merged_linear, self.p_node_conv3))
                y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
                
            node_output = torch.cat((cur_message_layer,y_cur_message_layer),axis = 0)
            #node_output = torch.nn.functional.normalize(node_output, p=2, dim=1)
            node_embedding.append(node_output)    
                    
        node_embedding_0 = node_embedding[0]
        node_embedding_1 = node_embedding[1]
        if self.embeddingMethod == 1:  # MRGNN
            # Always use compact indexing aligned with current active nodes (CPU numpy for downstream)
            compact_nodes = np.arange(node_embedding_0.size(0))
            embeds = [node_embedding_0, node_embedding_1]
            message_layer = torch.zeros(lay_num, len(compact_nodes), self.embedding_size, device=self.device)
            for l in range(lay_num):
                result_temp = self.layerNodeAttention_weight(embeds, compact_nodes, l)
                message_layer[l] = result_temp
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)
        # message_layer = torch.stack(node_embedding)          
        # cur_message_layer = message_layer[:, :nodes_cnt, :]
        # y_cur_message_layer = message_layer[:, nodes_cnt:, :]
        S_comm_global = None
        if community_index is not None and community_batch is not None and len(community_index) > 0:
            comm_emb = scatter_mean(cur_message_layer[0], community_index, dim=0)
            S_comm_global = scatter_mean(comm_emb, community_batch, dim=0)
        if S_comm_global is None:
            if node_batch is not None and len(node_batch) > 0 and node_batch[0].numel() > 0:
                S_comm_global = scatter_mean(cur_message_layer[0], node_batch[0], dim=0)
            else:
                S_comm_global = torch.mean(cur_message_layer[0], dim=0, keepdim=True)
        q_list = []
        w_layer = []
        for l in range(lay_num):
            y_potential = y_cur_message_layer[l]
            rep_y = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'].cuda(self.device),                                       rep_global[l]['m'], rep_global[l]['n'], y_potential.cuda(self.device))
            if node_batch is not None and len(node_batch) > l and node_batch[l].numel() > 0:
                nb = node_batch[l]
                num_graphs = int(torch.max(nb).item()) + 1
                S_virtual = scatter_mean(cur_message_layer[l], nb, dim=0)
                S_comm_graph = S_comm_global
                if S_comm_graph.dim() == 1:
                    S_comm_graph = S_comm_graph.unsqueeze(0).repeat(num_graphs, 1)
                elif S_comm_graph.size(0) < num_graphs:
                    pad = torch.zeros((num_graphs - S_comm_graph.size(0), self.embedding_size), device=self.device)
                    S_comm_graph = torch.cat([S_comm_graph, pad], dim=0)
                S_global = torch.concat([S_virtual, S_comm_graph], dim=1)
                S_global_nodes = S_global[nb]
                adv_all = self.adv_mlp(torch.concat([S_global_nodes, cur_message_layer[l]], dim=1))
                mean_adv_graph = scatter_mean(adv_all, nb, dim=0)
                q_on_all = self.value_mlp(S_global)[nb] + (adv_all - mean_adv_graph[nb])
            else:
                S_virtual = torch.mean(cur_message_layer[l], dim=0, keepdim=True)
                S_comm_use = S_comm_global if S_comm_global.dim() == 2 else S_comm_global.unsqueeze(0)
                S_global = torch.concat([S_virtual, S_comm_use], dim=1)
                S_global_nodes = S_global.repeat(cur_message_layer[l].size(0), 1)
                adv_all = self.adv_mlp(torch.concat([S_global_nodes, cur_message_layer[l]], dim=1))
                mean_adv_graph = torch.mean(adv_all, dim=0, keepdim=True)
                q_on_all = self.value_mlp(S_global) + (adv_all - mean_adv_graph)
            w_layer.append((self.act(rep_y @ self.w_layer1))@self.w_layer2)
            q_list.append(q_on_all)
        w_layer = torch.concat(w_layer,dim = 1)
        w_layer_softmax = F.softmax(w_layer,dim = 1)
        q = w_layer_softmax[:,0].unsqueeze(1) * q_list[0] + w_layer_softmax[:,1].unsqueeze(1) * q_list[1]
        return q
