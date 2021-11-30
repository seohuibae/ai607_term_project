import torch 

def construct_negative_graph(edge_index, num_nodes, k, device): # k: number of negative examples
    src, dst = edge_index[0,:], edge_index[1,:]
    neg_src = src.unsqueeze(1).repeat((1,k))
    neg_dst = torch.randint(0, num_nodes, (len(src)*k,)).unsqueeze(1).to(device)
    neg_edge_index = torch.cat([neg_src, neg_dst], dim=1).T
    # print(neg_edge_index.shape) # [2, 1000]
    return neg_edge_index  

def construct_gt_negative(edge_index, gt_negative, ratio, num_nodes, k, device):
    # Choose from the ground truth
    num_gt = int(edge_index.shape[1] * ratio) # number of gt ratio 
    num_gt = min(num_gt, len(gt_negative))
    neg_gt_idxs = torch.randint(0, len(gt_negative), (num_gt,)).to(device)
    gt_negative = [item.unsqueeze(-1) for item in gt_negative[neg_gt_idxs]]
    gt_negative = torch.cat(gt_negative, dim=1)

    num_neg = edge_index.shape[1] - num_gt
    # sampling 
    src = edge_index[0,:num_neg]
    neg_src = src.unsqueeze(1).repeat((1,k))
    neg_dst = torch.randint(0, num_nodes, (len(src)*k,)).unsqueeze(1).to(device)
    neg_edge_index = torch.cat([neg_src, neg_dst], dim=1).T
    neg_edge_index = torch.cat([neg_edge_index, gt_negative], dim=-1)
    return neg_edge_index


def metric(pos_pred, neg_pred):
    pos_label = 1 
    neg_label = 0 
    TP = len(pos_pred[pos_pred==1])
    FP = len(neg_pred[neg_pred==1])
    TN = len(neg_pred[neg_pred==0])
    FN = len(pos_pred[pos_pred==0])
    N = TP+FP+TN+FN
    accuracy = (TP+TN)/N
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score =  2*(recall * precision) / (recall + precision)
    
    return accuracy, precision, recall, f1_score