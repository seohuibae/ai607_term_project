import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data  import Data
from torch_geometric.data import DataLoader
import time

from loader import load_data
from models import *
from utils import *
from config import *
from evaluate import *


class Baseline(nn.Module):
    def __init__(self, x_s_dim, x_t_dim, emb_dim, gnn_hiddens, pred_emb_dim, **kwargs):
        super(Baseline, self).__init__(**kwargs)
        # emb_s = LinearEmbedding(input_dim=data.x_s.shape[1], output_dim=emb_dim)
        # emb_t = LinearEmbedding(input_dim=data.x_t.shape[1], output_dim=emb_dim)
        self.emb_s = nn.Embedding(x_s_dim, emb_dim)
        self.emb_t = nn.Embedding(x_t_dim, emb_dim)
        self.feature_extractor = GCN(input_dim=emb_dim, hiddens=gnn_hiddens, output_dim=pred_emb_dim)
        self.link_predictor = LinkPredictor(emb_dim=pred_emb_dim)

    def forward(self, x_s, x_t, edge_index, pos_edge_index, neg_edge_index, device):
        x_s = self.emb_s(x_s)
        x_t = self.emb_s(x_t)
        x_s, x_t = self.feature_extractor(x_s=x_s, x_t=x_t, edge_index=edge_index)
        loss, pos_score, neg_score = self.link_predictor(x_t, pos_edge_index, neg_edge_index)

        return loss, pos_score, neg_score


def main(trials=1):
    data, train_true_samples, train_false_samples, valid_true_samples, valid_false_samples, query_samples = load_data(device)
    edge_index = data.edge_index.to(device)
    x_s=data.x_s.to(device)
    x_t=data.x_t.to(device)

    print("experiment: ", args.expname)
    accs = []
    for t in range(trials):
        model = Baseline(x_s_dim=data.N_s, x_t_dim=data.N_t, emb_dim=args.emb_dim, gnn_hiddens=args.hiddens, pred_emb_dim=args.pred_emb_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        best_epoch = 0
        curr_step = 0
        best_val_acc = 0
        best_val_loss = 10000
        best_val_acc_trail=0

        model.train()
        for epoch in range(args.epochs):
            begin = time.time()
            pos_edge_index = train_true_samples.t().to(device)
            neg_edge_index = construct_negative_graph(pos_edge_index, data.N_t, 1, device)
            # neg_edge_index = construct_gt_negative(pos_edge_index, train_false_samples, 0.9, data.N_t, 1, device)

            loss, pos_score, neg_score  = model(x_s, x_t, edge_index, pos_edge_index, neg_edge_index, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pos_edge_index = valid_true_samples.t().to(device)
                neg_edge_index = valid_false_samples.t().to(device)

                val_loss, pos_score, neg_score  = model(x_s, x_t, edge_index, pos_edge_index, neg_edge_index, device)
                pos_pred, neg_pred, thr = predict(pos_score, neg_score)
                accuracy, precision, recall, f1_score = metric(pos_pred, neg_pred)
                val_acc = accuracy

            if val_acc > best_val_acc and epoch > 100:
                curr_step = 0
                best_epoch = epoch
                best_val_acc = val_acc
                best_val_loss= val_loss.item()
                if val_acc>best_val_acc_trail:
                    best_val_acc_trail = val_acc
            else:
                curr_step +=1

            # print(f"epoch={epoch+1}, train_loss={loss.item():.5f}, val_loss={val_loss.item():.5f}, val_acc={accuracy:.5f}, prec={precision:.5f}, recall={recall:.5f}, f1={f1_score:.5f}, best_val_acc_trail={best_val_acc_trail:.5f}, time={(time.time()-begin):.5f}s")
            print(f"epoch={epoch+1}, train_loss={loss.item():.5f}, val_loss={val_loss.item():.5f}, val_acc={accuracy:.5f}, prec={precision:.5f}, recall={recall:.5f}, f1={f1_score:.5f}, best_val_acc={best_val_acc:.5f}, best_val_acc_trail={best_val_acc_trail:.5f}, time={(time.time()-begin):.5f}s")
            if curr_step > args.early_stop:
                break
        accs.append(best_val_acc_trail)

    save_data(edge_index, pos_edge_index, neg_edge_index, pos_pred, neg_pred)
    return accs

if __name__ == "__main__":
    accs = main(1)
    print("experiment: ", args.expname)
    print(f"result: {np.mean(accs):.5f}({np.var(accs):.5f})")



