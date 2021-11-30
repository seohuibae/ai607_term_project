import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from config import *
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv, SAGEConv

from itertools import combinations

def create_paper_edge_index(edge_index, num_authors): 
    
    src = edge_index[0]
    tgt = edge_index[1]
    # print(edge_index.shape) # ([2, 1142106])

    paper_edge_index = []
    for tgt_idx in range(num_authors):
        paper_neighbors = src[(tgt==tgt_idx)].cpu().tolist()
        es = list(combinations(paper_neighbors, 2))
        es = map(list, es)
        paper_edge_index.extend(es)
        # break
    paper_edge_index = torch.LongTensor(paper_edge_index).t()
    src, tgt = paper_edge_index[0], paper_edge_index[1]
    # paper_edge_index_ = torch.cat([tgt.unsqueeze(0),src.unsqueeze(0)], dim=0)
    # paper_edge_index = torch.cat([paper_edge_index, paper_edge_index_], dim=1)
    print(paper_edge_index.shape) # ([2, 70531664])*2

    num_gen_edges = paper_edge_index.size(1)
    perm = torch.randperm(num_gen_edges)
    idx = perm[:int(num_gen_edges*0.1)]
    paper_edge_index = paper_edge_index[:,idx]
        
    return paper_edge_index
            

class BiLevelGraphConvolution(nn.Module):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 activation=nn.ReLU(),
                 bias=False,
                 featureless=False, **kwargs):
        super(BiLevelGraphConvolution, self).__init__(**kwargs)
        self.type = 'gcn'
        self.dropout = dropout
        self.activation = activation
        self.featureless = featureless
        self.usebias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin_s = nn.Linear(input_dim, input_dim)
        self.lin_t = nn.Linear(input_dim, input_dim)
        self.conv_1 = GCNConv(input_dim, output_dim, bias=bias)
        self.conv_2 = GCNConv(output_dim, output_dim, bias=bias)

    def forward(self, edge_index, paper_edge_index, x_s, x_t, training=None):
        # edge_index: src(papers) , tgts(authors)
        N_s = x_s.shape[0]
        N_t = x_t.shape[0]
        # dropout
        if training:
            x_s = F.dropout(x_s, self.dropout)
            x_t = F.dropout(x_t, self.dropout)

        # 0. to different embedding space
        x_s = self.lin_s(x_s)
        x_t = self.lin_s(x_t)

        # 1. level 1 convolve: authors -> papers 
        # concatenate
        x = torch.cat([x_s, x_t], dim=0)
        src = edge_index[0]
        dst = edge_index[1]
        dst = torch.add(dst, N_s) # reindexed 
        # change direction authors to papers ? 
        # src_ = dst
        # dst_ = src 
        edge_index_reindexed = torch.cat([src.unsqueeze(0),dst.unsqueeze(0)], dim=0)
        edge_index_reindexed_T  = torch.cat([dst.unsqueeze(0),src.unsqueeze(0)], dim=0)
        edge_index_reindexed = torch.cat([edge_index_reindexed, edge_index_reindexed_T], dim=1)
        # print(edge_index_reindexed.shape)

        x = self.conv_1(x, edge_index_reindexed, edge_weight=None) # convolve
        x = self.activation(x) # activation
        x_s = x[:N_s]
        x_t = x[N_s:]
        
        # 2. level 2 convolve: papers -> papers 
        x_s = self.conv_2(x_s, paper_edge_index, edge_weight=None) # convolve
        x_s = self.activation(x_s) # activation

       
        return x_s, x_t


class BiLevelGCN(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim,**kwargs):
        super(BiLevelGCN, self).__init__(**kwargs)
        
        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        self.layers_ = nn.ModuleList([])

        layer0 = BiLevelGraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0], dropout=0,
                                  activation=nn.ReLU())
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], dropout=args.dropout,
                                      activation=nn.ReLU())
            self.layers_.append(layertemp)

        layer_1 = BiLevelGraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim, dropout=args.dropout,
                                            activation=lambda x: x)

        self.layers_.append(layer_1)
        self.hiddens = hiddens


    def forward(self, edge_index, paper_edge_index, x_s, x_t, training=None):

        for layer in self.layers_:
            x_s, x_t = layer(edge_index, paper_edge_index, x_s, x_t, training=training)
        return x_s, x_t
