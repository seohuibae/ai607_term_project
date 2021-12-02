import torch 

def predict(pos_score, neg_score):
    thr = torch.mean(torch.cat([pos_score, neg_score], dim=0))

    pos_pred = torch.zeros((len(pos_score),))
    neg_pred = torch.zeros((len(neg_score),))

    pos_indices = torch.tensor([i for i in range(len(pos_score))])
    neg_indices = torch.tensor([i for i in range(len(neg_score))])

    pos_correct = pos_indices[(pos_score>=thr).squeeze(1).to(torch.bool)]
    neg_correct = neg_indices[(neg_score<thr).squeeze(1).to(torch.bool)]

    pos_pred[pos_correct] = 1
    neg_pred[neg_correct] = 1

    return pos_pred, neg_pred 

