import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from config import *
from utils import *
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv, SAGEConv

from itertools import combinations
from itertools import permutations

# def constrain_neighbors_of_hubs(edge_index): 
#     degree_dict = degree_dict_authors(edge_index)

#     return edge_index 
class BiLevelDropGraphConvolution(nn.Module):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 activation=nn.ReLU(),
                 bias=False,
                 featureless=False, **kwargs):
        super(BiLevelDropGraphConvolution, self).__init__(**kwargs)
        self.type = 'gcn'
        self.dropout = dropout
        self.activation = activation
        self.featureless = featureless
        self.usebias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin_s = nn.Linear(input_dim, input_dim)  # s to space t
        self.lin_t = nn.Linear(input_dim, input_dim)  # t to space s

        # self.conv_1 = GCNConv(input_dim, output_dim, bias=bias)
        # self.conv_2 = GCNConv(input_dim, output_dim, bias=bias)
        self.conv_1 = GATConv(input_dim, output_dim, bias=bias)
        self.conv_2 = GATConv(input_dim, output_dim, bias=bias)
        # self.conv_1 = SAGEConv(input_dim, output_dim, bias=bias)
        # self.conv_2 = SAGEConv(input_dim, output_dim, bias=bias)

    def forward(self, edge_index, paper_edge_index, author_edge_index, x_s, x_t,training=None):

        degree_dict = degree_dict_authors(edge_index)
        # for k,v in enumerate(degree_dict): 
        #     degree_dict[k] = 1/v # inverse

        mask = torch.zeros(len(edge_index[0])).bool()
        # degree_dict[i] for i in range(len(edge_index[1]))
        # p = torch.zeros((len(edge_index[1]),))
        # for i in range(len(edge_index[1])):
        #     aid = edge_index[1][i]
        #     d = degree_dict[aid]
        #     p[i] = nn.Sigmoid()(1/d) # probability to be sampled
        # torch.sample()

        # too long         
        # mask = torch.ones(len(edge_index[0])).bool()
        # max_deg = max(list(degree_dict.values()))
        # T = 1.0
        # drop_percentage = 0
        # print(x_t.size(0))
        # for aid in range(x_t.size(0)):
        #     aid_deg = degree_dict[aid]
        #     if aid_deg > 80:
        #         edge_index_indices = torch.tensor([i for i in range(len(edge_index[0]))])
        #         aid_drop_e_indices = edge_index_indices[aid==edge_index[1]]
        #         ratio = (aid_deg/max_deg)/T
        #         # print(ratio)
        #         num_drop_neighbors = int(len(aid_drop_e_indices)*ratio)
        #         aid_drop_e_indices = np.random.choice(aid_drop_e_indices.cpu().detach().numpy(), num_drop_neighbors, replace=False) # which to drop 
        #         mask[aid_drop_e_indices] = 0. 
        #         drop_percentage += len(aid_drop_e_indices)
        # drop_percentage /= len(edge_index[0])
        # edge_index = masked_select(edge_index, mask.unsqueeze(0).repeat(2,1))

        print(drop_percentage)
        print(edge_index.shape)
        input()

        # edge_index: src(papers) , tgts(authors)
        N_s = x_s.shape[0]
        N_t = x_t.shape[0]
        # dropout
        if training:
            x_s = F.dropout(x_s, self.dropout)
            x_t = F.dropout(x_t, self.dropout)

        # 0. to different embedding space
        x_s = self.lin_s(x_s)
        x_t = self.lin_t(x_t)

        # 1. level 1 convolve: authors -> papers
        # concatenate
        src = edge_index[0]
        dst = edge_index[1]
        dst = torch.add(dst, N_s) # reindexed
        # change direction authors to papers ?
        # src_ = dst
        # dst_ = src
        edge_index_reindexed = torch.cat([src.unsqueeze(0),dst.unsqueeze(0)], dim=0) # TODO 
        edge_index_reindexed_T  = torch.cat([dst.unsqueeze(0),src.unsqueeze(0)], dim=0)
        #edge_index_reindexed = torch.cat([edge_index_reindexed, edge_index_reindexed_T], dim=1)
        # print(edge_index_reindexed.shape)

        x = torch.cat([x_s, x_t], dim=0)

        edge = torch.cat([edge_index_reindexed, author_edge_index], dim=1)
        new_x_t = self.conv_1(x, edge) # convolve
        new_x_t = self.activation(new_x_t) # activation

        edge = torch.cat([edge_index_reindexed_T, paper_edge_index], dim=1)
        new_x_s = self.conv_2(x, edge) # convolve
        new_x_s = self.activation(new_x_s) # activation

        x_s = new_x_s[:N_s]
        x_t = new_x_t[N_s:]

        return x_s, x_t


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
        self.lin_s = nn.Linear(input_dim, input_dim)  # s to space t
        self.lin_t = nn.Linear(input_dim, input_dim)  # t to space s

        # self.conv_1 = GCNConv(input_dim, output_dim, bias=bias)
        # self.conv_2 = GCNConv(input_dim, output_dim, bias=bias)
        self.conv_1 = GATConv(input_dim, output_dim, bias=bias)
        self.conv_2 = GATConv(input_dim, output_dim, bias=bias)
        # self.conv_1 = SAGEConv(input_dim, output_dim, bias=bias)
        # self.conv_2 = SAGEConv(input_dim, output_dim, bias=bias)

    def forward(self, edge_index, paper_edge_index, author_edge_index, x_s, x_t, training=None):
        # edge_index: src(papers) , tgts(authors)
        N_s = x_s.shape[0]
        N_t = x_t.shape[0]
        # dropout
        if training:
            x_s = F.dropout(x_s, self.dropout)
            x_t = F.dropout(x_t, self.dropout)

        # 0. to different embedding space
        x_s = self.lin_s(x_s)
        x_t = self.lin_t(x_t)

        # 1. level 1 convolve: authors -> papers
        # concatenate
        src = edge_index[0]
        dst = edge_index[1]
        dst = torch.add(dst, N_s) # reindexed
        # change direction authors to papers ?
        # src_ = dst
        # dst_ = src
        edge_index_reindexed = torch.cat([src.unsqueeze(0),dst.unsqueeze(0)], dim=0)
        edge_index_reindexed_T  = torch.cat([dst.unsqueeze(0),src.unsqueeze(0)], dim=0)
        #edge_index_reindexed = torch.cat([edge_index_reindexed, edge_index_reindexed_T], dim=1)
        # print(edge_index_reindexed.shape)

        x = torch.cat([x_s, x_t], dim=0)

        edge = torch.cat([edge_index_reindexed, author_edge_index], dim=1)
        new_x_t = self.conv_1(x, edge) # convolve
        new_x_t = self.activation(new_x_t) # activation

        edge = torch.cat([edge_index_reindexed_T, paper_edge_index], dim=1)
        new_x_s = self.conv_2(x, edge) # convolve
        new_x_s = self.activation(new_x_s) # activation

        x_s = new_x_s[:N_s]
        x_t = new_x_t[N_s:]

        return x_s, x_t


class BiLevelDropGCN(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim,**kwargs):
        super(BiLevelDropGCN, self).__init__(**kwargs)

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        self.layers_ = nn.ModuleList([])

        layer0 = BiLevelDropGraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0], dropout=0,
                                  activation=nn.ReLU())
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = BiLevelDropGraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], dropout=args.dropout,
                                      activation=nn.ReLU())
            self.layers_.append(layertemp)

        layer_1 = BiLevelDropGraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim, dropout=args.dropout,
                                            activation=lambda x: x)

        self.layers_.append(layer_1)
        self.hiddens = hiddens


    def forward(self, edge_index, paper_edge_index, author_edge_index, x_s, x_t, training=None):
        for layer in self.layers_:
            x_s, x_t = layer(edge_index, paper_edge_index, author_edge_index, x_s, x_t, training=training)
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
            layertemp = BiLevelGraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], dropout=args.dropout,
                                      activation=nn.ReLU())
            self.layers_.append(layertemp)

        layer_1 = BiLevelGraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim, dropout=args.dropout,
                                            activation=lambda x: x)

        self.layers_.append(layer_1)
        self.hiddens = hiddens


    def forward(self, edge_index, paper_edge_index, author_edge_index, x_s, x_t, training=None):

        for layer in self.layers_:
            x_s, x_t = layer(edge_index, paper_edge_index, author_edge_index, x_s, x_t, training=training)
        return x_s, x_t
