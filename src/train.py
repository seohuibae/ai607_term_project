import torch 
import torch.nn as nn 
import numpy as np 
from torch_geometric.data  import Data
from torch_geometric.data import DataLoader 

from loader import load_data
from models import *
from utils import * 
from config import *


class Model(nn.Module):
    def __init__(self, x_s_dim, x_t_dim, emb_dim, gnn_hiddens, pred_emb_dim, **kwargs):
        super(Model, self).__init__(**kwargs)
        # emb_s = LinearEmbedding(input_dim=data.x_s.shape[1], output_dim=emb_dim)
        # emb_t = LinearEmbedding(input_dim=data.x_t.shape[1], output_dim=emb_dim)
        self.emb_s = nn.Embedding(x_s_dim, emb_dim)
        self.emb_t = nn.Embedding(x_t_dim, emb_dim)
        self.feature_extractor = GCN(input_dim=emb_dim, hiddens=gnn_hiddens, output_dim=pred_emb_dim)
        self.link_predictor = LinkPredictor(emb_dim=pred_emb_dim)

    def forward(self, data, pos_edge_index, neg_edge_index, device):
        x_s, x_t, edge_index = data.x_s.to(device), data.x_t.to(device), data.edge_index.to(device)
        x_s = self.emb_s(x_s)
        x_t = self.emb_s(x_t) 
        x_s, x_t = self.feature_extractor(x_s=x_s, x_t=x_t, edge_index=edge_index)
        loss, pos_score, neg_score = self.link_predictor(x_t, pos_edge_index, neg_edge_index)

        return loss, pos_score, neg_score
    
def pred(pos_score, neg_score):
    thr = torch.mean(torch.cat([pos_score, neg_score], dim=0))
    pos_indices = [i for i in range(len(pos_score))]
    neg_indices = [i for i in range(len(neg_score))]
    pos_pred = pos_indices[pos_score>=thr]
    neg_pred = neg_indices[neg_score<thr]

    return pos_pred, neg_pred 

def main(): 

    data, train_true_samples, train_false_samples, valid_true_samples, valid_false_samples, query_samples = load_data()
    data
    model = Model(x_s_dim=data.N_s, x_t_dim=data.N_t, emb_dim=args.emb_dim, gnn_hiddens=args.hiddens, pred_emb_dim=args.pred_emb_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   

    best_epoch = 0
    curr_step = 0
    best_val_acc = 0

    model.train() 
    for epoch in range(args.epochs):
        neg_edge_index = construct_negative_graph(train_true_samples.to(device), data.N_t, 1, device)
        
        loss, pos_score, neg_score  = model(data, train_true_samples,neg_edge_index, device)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        pos_pred, neg_pred = pred(pos_score, neg_score)

        pos_acc = len(pos_pred[pos_pred==1])/len(pos_pred)
        neg_acc = len(neg_pred[neg_pred==0])/len(neg_pred)
        # print(f"pos acc: {pos_acc:.5f}, neg_acc: {neg_acc:.5f}")

    # return best_test_acc, best_test_auc, best_test_f1

if __name__ == "__main__":
    main()

    