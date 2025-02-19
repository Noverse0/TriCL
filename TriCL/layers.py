from typing import Optional, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class ProposedConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, 
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False, 
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        # self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None) 
            self.register_parameter('bias_e2n', None) 
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None
    
    def forward(self, x: Tensor, hyperedge_index: Tensor, 
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None, dropout_rate: Optional[float] = 0.0):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]
                
            else:
                De_inv = De.pow(-1.0)
                De_inv[De_inv == float('inf')] = 0
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                
                norm_n2e = De_inv[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_e2n = Dn_inv_sqrt[node_idx]


            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        
        x = self.lin_n2e(x)
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e, 
                               size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=dropout_rate, training=self.training)
        
        x = self.lin_e2n(e)
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n, 
                               size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n
        n = self.act(n)
        return n, e # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j
    
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight1 = Parameter(torch.Tensor(in_ft, out_ft))
        self.weight2 = Parameter(torch.Tensor(out_ft, out_ft))
        self.prelu_e = nn.PReLU()
        self.prelu_n = nn.PReLU()
        self.bias = bias
        if self.bias:
            self.bias1 = Parameter(torch.Tensor(out_ft))
            self.bias2 = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, x: torch.Tensor, Q: torch.Tensor, P: torch.Tensor, dropout_rate: Optional[float] = 0.0):
        e = F.dropout(x, p=dropout_rate, training=self.training)
        e = e.matmul(self.weight1)
        e = Q.matmul(e)
        if self.bias:
            e = e + self.bias1
        e = self.prelu_e(e)
        
        n = F.dropout(e, p=dropout_rate, training=self.training)
        n = n.matmul(self.weight2)
        n = P.matmul(n)
        if self.bias:
            n = n + self.bias2
        n = self.prelu_n(n)
        return n, e
