import torch 
import torch.nn as nn
import torch.nn.functional as F  
from torch.nn import init
import math 
from config import *

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv, SAGEConv

class GraphConvolution(nn.Module):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 activation=nn.ReLU(),
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.type = 'gcn'

        self.dropout = dropout
        self.activation = activation
        self.featureless = featureless
        self.usebias = bias
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.conv = GCNConv(input_dim, output_dim, bias=bias)

    def forward(self, edge_index, x_s, x_t, training=None):
        N_s = x_s.shape[0]
        N_t = x_t.shape[0]
        # dropout
        if training is not False:
            x_s = F.dropout(x_s, self.dropout)
            x_t = F.dropout(x_t, self.dropout)
        
        # concatenate 
        x = torch.cat([x_s, x_t], dim=0)
        edge_index[1] += N_s 
        # convolve
        x = self.conv(x, edge_index, edge_weight=None) # external library
        
        # activation
        x = self.activation(x)
        x_s = x[:N_s]
        x_t = x[N_s:]

        return x_s, x_t 

class LinearEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim,
                 activation=nn.ReLU(),
                 bias=False,**kwargs):
        super(LinearEmbedding, self).__init__(**kwargs)
        self.type = 'linear'

        self.activation = activation
        self.usebias = bias
        self.lin = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        # dropout
        if training is not False:
            x = F.dropout(x, self.dropout)
        # projection
        x = self.lin(x)
        # activation
        x = self.activation(x)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)
        
        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        self.layers_ = nn.ModuleList([])

        layer0 = GraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0], dropout=0,
                                  activation=nn.ReLU())
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_], dropout=args.dropout,
                                      activation=nn.ReLU())
            self.layers_.append(layertemp)

        layer_1 = GraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim, dropout=args.dropout,
                                            activation=lambda x: x)

        self.layers_.append(layer_1)
        self.hiddens = hiddens


    def forward(self, edge_index, x_s, x_t, training=None):

        for layer in self.layers_:
            x_s, x_t = layer(edge_index, x_s, x_t, training=training)
        return x_s, x_t


class LinkPredictor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim 
        self.linear_src = nn.Linear(emb_dim, emb_dim) 
        self.linear_dst = nn.Linear(emb_dim, emb_dim)
        self.bilinear = nn.Bilinear(emb_dim, emb_dim, 1)
        # self.fc = nn.Linear(emb_dim, num_classes) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, pos_edge_index, neg_edge_index=None): # x: gnn output

        pos_score = self.bilinear(self.linear_src(x[pos_edge_index[0,:]]), self.linear_dst(x[pos_edge_index[1,:]]))
        pos_score = self.sigmoid(pos_score)
        neg_score = self.bilinear(self.linear_src(x[neg_edge_index[0,:]]), self.linear_dst(x[neg_edge_index[1,:]]))
        neg_score = self.sigmoid(neg_score)

        loss = self._compute_link_loss(pos_score, neg_score)
        return loss, pos_score, neg_score

    def _compute_link_loss(self, pos_score, neg_score):            
        margin=1
        n_edges = pos_score.shape[0]
        if weights is None: 
            loss = ((margin-pos_score).unsqueeze(1) + (neg_score.view(n_edges, -1))).clamp(min=0).mean()

        return loss 