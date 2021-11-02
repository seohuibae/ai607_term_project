import torch 

def construct_negative_graph(edge_index, num_nodes, k, device): # k: number of negative examples
    src, dst = edge_index[0,:], edge_index[1,:]
    neg_src = src.unsqueeze(1).repeat((1,k))
    neg_dst = torch.randint(0, num_nodes, (len(src)*k,)).unsqueeze(1).to(device)
    neg_edge_index = torch.cat([neg_src, neg_dst], dim=1).T
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