from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
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
                 embeddingMethod=1, aux_dim=4, device=None, node_attr=False):
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
        self.act = nn.ReLU()
        
        # [2, embed_dim] -> [3, embed_dim] for HCA features
        self.w_n2l = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(3, self.embedding_size)))

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
                                                                             size=(self.reg_hidden + self.aux_dim, 1)))
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
            self.last_w = self.h1_weight

        ## [embed_dim, 1]
        # cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)
        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.embedding_size, 1)))
        #self.w_layer = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
        #                                                             #size=(embedding_size, 1)))
        self.w_layer1 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(embedding_size, 128)))
        self.w_layer2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(128, 1)))
        
        # HCA - Macro Level GCN Weights
        # [embed_dim, embed_dim]
        self.w_macro = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                       size=(self.embedding_size, self.embedding_size)))
        
        # HCA - Decoder Weights
        # Community Score MLP: [embed_dim + embed_dim, 1] (h_comm || h_global) -> Score
        # h_global is [batch_size, embed_dim] or [1, embed_dim]?
        # In this code, global rep seems to be 'cur_message_layer' aggregated?
        # Actually rep_global maps Nodes -> Global.
        # We will define a scoring weight.
        self.w_comm_score = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                            size=(2 * self.embedding_size, 1)))

        # Micro Score MLP: [embed_dim + embed_dim, 1] (h_u || h_comm) -> Q
        self.w_micro_score = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(2 * self.embedding_size, 1)))
        
        self.flag = 0

    def train_forward(self, node_input, subgsum_param, n2nsum_param, action_select, aux_input, adj, v_adj, comm_adj):
        
        nodes_cnt = n2nsum_param[0]['m']
        y_nodes_size = subgsum_param[0]['m']
        
        

        
        # HCA Features Projection
        # node_input is [BatchNodes, 3] from PrepareBatchGraph (it aggregates all unique nodes)
        # We assume node_input corresponds to the nodes indexed in n2nsum_param matrices.
        
        # Projection: [BatchNodes, 3] -> [BatchNodes, 64]
        message_input = torch.matmul(node_input, self.w_n2l)
        message_input = self.act(message_input)
        
        # Repeat for Layer 0 and Layer 1 (Multiplex structure)
        lay_num = 2
        
        # Calculate max community size for padding
        max_comm_size = 0
        for l in range(lay_num):
             if subgsum_param[l]['m'] > max_comm_size:
                 max_comm_size = subgsum_param[l]['m']
        
        cur_message_layer_all = message_input.unsqueeze(0).repeat(lay_num, 1, 1) # [2, nodes_cnt, 64]
        cur_message_layer_all = torch.nn.functional.normalize(cur_message_layer_all, p=2, dim=2)
        
        # Community Initialization
        # Use lists to allow different number of communities per layer
        y_cur_message_layer_all = []
        for l in range(lay_num):
             m_l = subgsum_param[l]['m']
             y_node_input_l = torch.ones((m_l, 3)).to(self.device)
             # Projection: [BatchComms, 3] -> [BatchComms, 64]
             y_message_input_l = torch.matmul(y_node_input_l, self.w_n2l)
             y_message_input_l = self.act(y_message_input_l)
             y_message_input_l = torch.nn.functional.normalize(y_message_input_l, p=2, dim=1)
             y_cur_message_layer_all.append(y_message_input_l)
        
        node_embedding = []
        lay_num = 2
        
        for l in range(lay_num):
            cur_message_layer = cur_message_layer_all[l]
            y_cur_message_layer = y_cur_message_layer_all[l]
             
            # Cross-Layer Heterogeneity Bias (Micro-Level Attention)
            other_l = 1 - l
            f_het = node_input[:, 0].unsqueeze(1) # [Nodes, 1]
            cross_layer_signal = cur_message_layer_all[other_l]
            cur_message_layer = cur_message_layer + 5.0 * f_het * cross_layer_signal
            
            lv = 0
            while lv < self.max_bp_iter:
                lv += 1
                
                # 1. Micro Aggregation
                n2npool = torch_sparse.spmm(n2nsum_param[l]['index'], n2nsum_param[l]['value'],\
                        n2nsum_param[l]['m'], n2nsum_param[l]['n'], cur_message_layer)
                node_linear = torch.matmul(n2npool, self.p_node_conv)
                
                # 2. Meso Aggregation (Weighted Pooling)
                y_n2npool = torch_sparse.spmm(subgsum_param[l]['index'], subgsum_param[l]['value'],\
                        subgsum_param[l]['m'], subgsum_param[l]['n'], cur_message_layer)
                y_node_linear = torch.matmul(y_n2npool, self.p_node_conv)
                
                # Node Update
                cur_message_layer_linear = torch.matmul(cur_message_layer, self.p_node_conv2)
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                cur_message_layer = self.act(torch.matmul(merged_linear, self.p_node_conv3))
                cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
                
                # Community Update
                y_cur_message_layer_linear = torch.matmul(y_cur_message_layer, self.p_node_conv2)
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_cur_message_layer = self.act(torch.matmul(y_merged_linear, self.p_node_conv3))
                y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            
            # 3. Macro Interaction (GCN)
            comm_agg = torch_sparse.spmm(comm_adj[l]['index'], comm_adj[l]['value'], \
                                         comm_adj[l]['m'], comm_adj[l]['n'], y_cur_message_layer)
            y_cur_message_layer = self.act(torch.matmul(comm_agg, self.w_macro))
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            
            node_output = torch.cat((cur_message_layer, y_cur_message_layer), axis=0) # [Nodes+Comms, Dim]
            
            # Pad if necessary for MRGNN Attention compatibility
            current_comm_size = subgsum_param[l]['m']
            if current_comm_size < max_comm_size:
                padding = torch.zeros((max_comm_size - current_comm_size, self.embedding_size)).to(self.device)
                node_output = torch.cat((node_output, padding), axis=0)
                
            node_embedding.append(node_output)

        # Apply MRGNN Layer Node Attention if used
        if self.embeddingMethod == 1:
            nodes = np.array(list(range(nodes_cnt + max_comm_size)))
            embeds = [node_embedding[0], node_embedding[1]]
            message_layer = torch.zeros(lay_num, len(nodes), self.embedding_size).to(self.device)
            for l in range(lay_num):
                result_temp = self.layerNodeAttention_weight(embeds, nodes, l)
                message_layer[l] = result_temp
            
            # Note: MRGNN combines layers for final embedding? 
            # But here `message_layer` replaces `node_embedding`?
            # We preserve the list structure for decoding.
            for l in range(lay_num):
                node_embedding[l] = message_layer[l]


        q_list = []
        for l in range(lay_num):
            node_emb = node_embedding[l][:nodes_cnt, :]
            
            # Slice community embedding to valid entries (remove padding)
            num_comms = subgsum_param[l]['m']
            comm_emb = node_embedding[l][nodes_cnt : nodes_cnt+num_comms, :]
            
            # --- Decoder Phase (Divide and Conquer) ---
            
            # 1. Community Heatmap & Selection
            h_global = torch.mean(comm_emb, dim=0, keepdim=True)
            h_global_expanded = h_global.expand(comm_emb.size(0), -1)
            comm_score_input = torch.cat([comm_emb, h_global_expanded], dim=1)
            comm_scores = torch.matmul(comm_score_input, self.w_comm_score)
            
            # Masking (Top 30%)
            k_top = max(1, int(comm_emb.size(0) * 0.3))
            _, top_k_indices = torch.topk(comm_scores.squeeze(), k_top)
            comm_mask = torch.zeros_like(comm_scores)
            comm_mask[top_k_indices] = 1.0
            
            # Project to Nodes
            # NodeMask = SubgSum_T * CommMask.
            # subgsum_param maps Nodes to Comm (Logic: value is weight).
            # We assume non-zero entry means membership.
            # subgsum_param[l] dimensions: [NumComm, NumNode]
            
            subg_index = subgsum_param[l]['index']
            subg_index_T = subg_index[[1, 0]]
            
            node_mask = torch_sparse.spmm(
                subg_index_T, 
                subgsum_param[l]['value'], 
                subgsum_param[l]['n'], 
                subgsum_param[l]['m'], 
                comm_mask
            )
            
            # 2. Node Pinpointing
            # h_comm_broadcast = SubgSum_T @ comm_emb
            h_comm_broadcast = torch_sparse.spmm(
                subg_index_T,
                subgsum_param[l]['value'],
                subgsum_param[l]['n'],
                subgsum_param[l]['m'],
                comm_emb
            )
            
            node_q_input = torch.cat([node_emb, h_comm_broadcast], dim=1)
            q_raw = torch.matmul(node_q_input, self.w_micro_score)
            
            # Apply Mask
            q_final = q_raw.clone()
            unselected_mask = (node_mask == 0)
            q_final[unselected_mask] = -1e9
            
            q_list.append(q_final)

        # Layer Aggregation
        w_layer = []
        for l in range(lay_num):
             comm_emb = node_embedding[l][nodes_cnt:, :]
             mean_comm = torch.mean(comm_emb, dim=0, keepdim=True)
             w = self.act(torch.matmul(mean_comm, self.w_layer1))
             w = torch.matmul(w, self.w_layer2)
             w_layer.append(w)
             
        w_layer = torch.cat(w_layer, dim=1) 
        w_layer_softmax = F.softmax(w_layer, dim=1)
        
        # Combined Q-values for ALL nodes [TotalNodes, 1]
        q_all = w_layer_softmax[:, 0] * q_list[0] + w_layer_softmax[:, 1] * q_list[1]
        
        # Prepare returns for calc_loss compatibility
        # calc_loss expects list of [Nodes, Dim]
        cur_message_layer_loss = [node_embedding[0][:nodes_cnt], node_embedding[1][:nodes_cnt]]

        if action_select is None:
            # Inference Mode: Return Q-values for all nodes
            return q_all, cur_message_layer_loss

        # Gather Q-values for selected actions [BatchSize, 1]
        q_pred_list = []
        for l in range(lay_num):
             # q_list[l] is [Nodes, 1]
             # action_select[l] is [Batch, Nodes]
             # q_pred_l = ActionSelect @ Q_l
             q_pred_l = torch_sparse.spmm(action_select[l]['index'], action_select[l]['value'],
                                          action_select[l]['m'], action_select[l]['n'],
                                          q_list[l])
             q_pred_list.append(q_pred_l)
             
        q_pred = w_layer_softmax[:, 0].unsqueeze(1) * q_pred_list[0] + w_layer_softmax[:, 1].unsqueeze(1) * q_pred_list[1]
        
        return q_pred, cur_message_layer_loss

    def test_forward(self, node_input, subgsum_param, n2nsum_param, rep_global, aux_input, adj, v_adj, comm_adj):
        q, _ = self.train_forward(node_input, subgsum_param, n2nsum_param, None, aux_input, adj, v_adj, comm_adj)
        return q
