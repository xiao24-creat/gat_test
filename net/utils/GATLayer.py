import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 定义GAT层
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.kaiming_uniform_(self.W.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#         self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
#         nn.init.kaiming_uniform_(self.a.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, input, adj):
#         """
#         输入维度：
#         - input: (N, in_features, T, V)
#         - adj: (V, V)
#
#         输出维度：
#         - h: (N, out_features, T, V)
#         """
#         N, C, T, V = input.size()
#         input = input.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
#         input = input.view(-1, V, C)  # (N*T, V, C)
#         h = torch.matmul(input, self.W)  # (N*T, V, out_features)
#         N_T = h.size(0)
#
#         a_input = torch.cat([h.repeat(1, 1, V).view(N_T, V * V, -1), h.repeat(1, V, 1)], dim=2).view(N_T, V, -1, 2 * self.out_features)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (N*T, V, V)
#
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=2)  # (N*T, V, V)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)  # (N*T, V, out_features)
#
#         if self.concat:
#             h_prime = F.elu(h_prime)
#
#         h_prime = h_prime.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()  # (N, out_features, T, V)
#         return h_prime

class PartitionedGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True, partitions=3):
        super(PartitionedGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.partitions = partitions

        # 共享的特征变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.kaiming_uniform_(self.W.data, a=0, mode='fan_in', nonlinearity='leaky_relu')

        # 每个分区独立的注意力参数
        self.a = nn.ParameterList([
            nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
            for _ in range(partitions)
        ])
        for a in self.a:
            nn.init.kaiming_uniform_(a.data, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # 预计算缓冲区
        self.register_buffer('edge_index', None)
        self.register_buffer('edge_partition', None)

    def init_adjacency(self, adj):
        """ 初始化邻接矩阵和分区信息 """
        if adj.dim() == 3:  # spatial策略的多分区情况
            edge_indices = []
            edge_partitions = []

            for k in range(adj.size(0)):
                part_adj = adj[k]
                edge_index = torch.nonzero(part_adj, as_tuple=False).t()
                edge_indices.append(edge_index)
                edge_partitions.append(torch.full((edge_index.size(1),), k))

            self.edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
            self.edge_partition = torch.cat(edge_partitions)  # [total_edges]
        else:  # 单分区情况
            self.edge_index = torch.nonzero(adj, as_tuple=False).t()
            self.edge_partition = torch.zeros(self.edge_index.size(1))

    def forward(self, input, adj=None):
        """
        输入维度：
        - input: (N, in_features, T, V)
        - adj: 可选，如果不为None则重新初始化

        输出维度：
        - h: (N, out_features, T, V)
        """
        if adj is not None:
            self.init_adjacency(adj)

        N, C, T, V = input.size()
        input = input.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        input = input.view(-1, V, C)  # (N*T, V, C)
        h = torch.matmul(input, self.W)  # (N*T, V, out_features)
        N_T = h.size(0)

        # 获取边信息
        src, dst = self.edge_index
        h_src = h[:, src]  # (N*T, E, out_features)
        h_dst = h[:, dst]  # (N*T, E, out_features)

        # 计算分区注意力
        e = torch.zeros(N_T, src.size(0), device=h.device)
        for k in range(self.partitions):
            mask = (self.edge_partition == k)
            if mask.any():
                edge_att = torch.cat([h_src[:, mask], h_dst[:, mask]], dim=-1)
                e_k = self.leakyrelu(torch.matmul(edge_att, self.a[k])).squeeze(-1)
                e[:, mask] = e_k

        # 注意力归一化(按目标节点分组)
        alpha = torch.zeros_like(e)
        for v in range(V):
            mask = (dst == v)
            if mask.any():
                alpha[:, mask] = F.softmax(e[:, mask], dim=1)

        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # # 在计算完alpha后保存权重
        # self.last_alpha = alpha.detach()  # [N*T, E]

        # 稀疏聚合
        h_prime = torch.zeros_like(h)
        h_prime.scatter_add_(1,
                             dst.view(1, -1, 1).expand(N_T, -1, self.out_features),
                             alpha.unsqueeze(-1) * h_src)

        if self.concat:
            h_prime = F.elu(h_prime)

        h_prime = h_prime.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()
        return h_prime

    # def visualize_partition_attention(self, save_path=None):
    #     """可视化各分区的注意力权重分布
    #     Args:
    #         save_path: 图片保存路径，如果为None则直接显示
    #     """
    #     import matplotlib.pyplot as plt
    #     import os
    #
    #     if self.last_alpha is None:
    #         print("No attention weights to visualize. Run forward pass first.")
    #         return
    #
    #     plt.figure(figsize=(15, 5))
    #     for k in range(self.partitions):
    #         mask = (self.edge_partition == k)
    #         if mask.any():
    #             alpha_k = self.last_alpha[:, mask].cpu().numpy().flatten()
    #
    #             plt.subplot(1, self.partitions, k + 1)
    #             plt.hist(alpha_k, bins=20, range=(0, 1))
    #             plt.title(f'Partition {k} Attention')
    #             plt.xlabel('Weight Value')
    #             plt.ylabel('Frequency')
    #
    #     plt.tight_layout()
    #     if save_path:
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         plt.savefig(save_path)
    #         plt.close()
    #     else:
    #         plt.show()
