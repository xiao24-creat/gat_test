import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 定义GAT层
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        输入维度：
        - input: (N, in_features, T, V)
        - adj: (V, V)

        输出维度：
        - h: (N, out_features, T, V)
        """
        N, C, T, V = input.size()
        input = input.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        input = input.view(-1, V, C)  # (N*T, V, C)
        h = torch.matmul(input, self.W)  # (N*T, V, out_features)
        N_T = h.size(0)

        a_input = torch.cat([h.repeat(1, 1, V).view(N_T, V * V, -1), h.repeat(1, V, 1)], dim=2).view(N_T, V, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (N*T, V, V)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # (N*T, V, V)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # (N*T, V, out_features)

        if self.concat:
            h_prime = F.elu(h_prime)

        h_prime = h_prime.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()  # (N, out_features, T, V)
        return h_prime




