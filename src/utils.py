import torch 
import numpy as np 
import os 

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

def create_paper_edge_index(edge_index, num_authors):
    if os.path.exists('tmp/paper_edge_index.npy'):
        print('loading')
        paper_edge_index = np.load('tmp/paper_edge_index.npy')
        paper_edge_index = torch.LongTensor(paper_edge_index)
    else:
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
        # paper_edge_index = set(paper_edge_index)
        # paper_edge_index = list(map(list, paper_edge_index))
        paper_edge_index = torch.LongTensor(paper_edge_index).t()
        # src, tgt = paper_edge_index[0], paper_edge_index[1]
        # paper_edge_index_ = torch.cat([tgt.unsqueeze(0),src.unsqueeze(0)], dim=0)
        # paper_edge_index = torch.cat([paper_edge_index, paper_edge_index_], dim=1)
        np.save('tmp/paper_edge_index.npy', paper_edge_index.cpu().detach().numpy())
        print('saved')
    print(paper_edge_index.shape) # ([2, 70531664])*2
    num_gen_edges = paper_edge_index.size(1)
    perm = torch.randperm(num_gen_edges)
    idx = perm[:int(num_gen_edges*0.1)]
    paper_edge_index = paper_edge_index[:,idx]
    print(paper_edge_index.shape)

    return paper_edge_index


def create_author_edge_index(edge_index, num_papers):
    if os.path.exists('tmp/author_edge_index.npy'):
        print('loading')
        author_edge_index = np.load('tmp/author_edge_index.npy')
        author_edge_index = torch.LongTensor(author_edge_index)
    else:
        src = edge_index[0] #p
        tgt = edge_index[1] #a
        # print(edge_index.shape)

        author_edge_index = []
        for src_idx in range(num_papers):
            author_neighbors = tgt[(src==src_idx)].cpu().tolist()
            es = list(combinations(author_neighbors, 2))
            es = map(list, es)
            author_edge_index.extend(es)
            # break
        # author_edge_index = set(author_edge_index)
        # author_edge_index = list(map(list, author_edge_index))
        author_edge_index = torch.LongTensor(author_edge_index).t()
        # src, tgt = author_edge_index[0], author_edge_index[1]
        # author_edge_index_ = torch.cat([tgt.unsqueeze(0),src.unsqueeze(0)], dim=0)
        # author_edge_index = torch.cat([author_edge_index, author_edge_index_], dim=1)
        np.save('tmp/author_edge_index.npy', author_edge_index.cpu().detach().numpy())
        print('saved')
    print(author_edge_index.shape)
    # num_gen_edges = author_edge_index.size(1)
    # perm = torch.randperm(num_gen_edges)
    # idx = perm[:int(num_gen_edges*0.1)]
    # author_edge_index = author_edge_index[:,idx]

    return author_edge_index

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
    if precision+recall != 0: 
        f1_score =  2*(recall * precision) / (recall + precision)
    else: 
        f1_score = 0.
    
    return accuracy, precision, recall, f1_score
    

def save_data(edge_index, true_samples, false_samples, pos_pred, neg_pred):
    np.save('preds/edge_index.npy', edge_index.cpu().detach().numpy())
    np.save('preds/true_samples.npy', true_samples.cpu().detach().numpy())
    np.save('preds/false_samples.npy', false_samples.cpu().detach().numpy())
    np.save('preds/pos_pred.npy', pos_pred.cpu().detach().numpy())
    np.save('preds/neg_pred.npy', neg_pred.cpu().detach().numpy())
    # np.save('preds/fns_idx.npy', fns_idx)
    # np.save('preds/fps_idx.npy', fps_idx)
    print('saved')

def load_pred_data():
    edge_index = np.load('preds/edge_index.npy')
    true_samples = np.load('preds/true_samples.npy')
    false_samples = np.load('preds/false_samples.npy')
    pos_pred = np.load('preds/pos_pred.npy')
    neg_pred = np.load('preds/neg_pred.npy')
    # fns_idx = np.load('preds/fns_idx.npy')
    # fps_idx = np.load('preds/fps_idx.npy')
    print('loaded')
    return edge_index, true_samples, false_samples, pos_pred, neg_pred