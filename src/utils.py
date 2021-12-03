import torch 
import torch.nn as nn 
import torch.nn.functional as F 
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

def degree_dict_authors(edge_index):
    from collections import Counter
    deg = Counter(edge_index[1].tolist())
    return deg

def degree_dict_papers(edge_index):
    from collections import Counter
    deg = Counter(edge_index[0].tolist())
    return deg

def sampling_uniform(paper_edge_index, ratio):
    num_gen_edges = paper_edge_index.size(1)
    perm = torch.randperm(num_gen_edges)
    idx = perm[:int(num_gen_edges*ratio)]
    paper_edge_index = paper_edge_index[:,idx]
    return paper_edge_index

def sampling_inv_to_degree_of_neighbors(edge_index, paper_edge_index, num_papers, num_authors, ratio):
    num_paper_edges= len(paper_edge_index[0])
    paper_edge_index_indices = torch.tensor([i for i in range(num_paper_edges)]) # initialize

    author_incents = torch.ones((num_authors,))
    v = torch.ones((len(edge_index[0])))
    edge_index_sparse = torch.sparse_coo_tensor(edge_index.cpu(), v, [num_papers, num_authors])
    
    # print(author_incents.shape)
    # print(edge_index_sparse)
    # print(edge_index_sparse.shape)
    paper_incents = torch.sparse.mm(edge_index_sparse, author_incents.unsqueeze(1)).squeeze(1)
    # print(paper_incents)
    # print(paper_incents.shape)
    # print(len(set(paper_incents.tolist())))
    # input()

    paper_edge_index_incents = torch.cat([paper_incents[paper_edge_index[0]].unsqueeze(0), paper_incents[paper_edge_index[1]].unsqueeze(0)], dim=0)
    # print(paper_edge_index_incents.shape)
    
    paper_edge_index_incents = torch.sum(paper_edge_index_incents, 0)

    

    # top_k = 200
    # v,i = torch.topk(paper_edge_index_incents, top_k)
    # paper_edge_index_tmp = torch.zeros((num_paper_edges,))
    # print(v)
    # paper_edge_index_tmp[i] = v
    # paper_edge_index_incents = paper_edge_index_tmp 
    # print(paper_edge_index_incents)
    # input() 
    
    # inverse 
    max_ = torch.max(paper_edge_index_incents)
    paper_edge_index_incents = max_ - paper_edge_index_incents
    print(paper_edge_index_incents)
    # print(paper_edge_index_incents.shape)
    # norm = torch.sum(paper_edge_index_incents)
    # prob = paper_edge_index_incents / norm
    # T = 0.95
    T = 1.8
    print('T', str(T))
    output = paper_edge_index_incents/T
    prob = F.log_softmax(output)
    prob = torch.exp(prob)
    print(torch.mean(prob))
    print(torch.min(prob))
    print(prob)
    # input()

    thr = 0
    print(len(paper_edge_index_indices))
    paper_edge_index_indices = torch.masked_select(paper_edge_index_indices, (prob>thr))
    print(len(paper_edge_index_indices))
    # input()

    paper_edge_index_indices = np.random.choice(paper_edge_index_indices.tolist(), int(num_paper_edges*ratio))
    paper_edge_index = paper_edge_index[:,paper_edge_index_indices]

    del author_incents
    del paper_incents 
    del paper_edge_index_incents
    del prob 
    del paper_edge_index_indices 
    del edge_index_sparse
    del v

    return paper_edge_index

def sampling_mixed(edge_index, paper_edge_index, num_papers, num_authors, ratio):
    paper_edge_index_uni = sampling_uniform(paper_edge_index, ratio*0.8)
    paper_edge_index_inv = sampling_inv_to_degree_of_neighbors(edge_index, paper_edge_index, num_papers, num_authors, ratio*0.2)
    paper_edge_index = torch.cat([paper_edge_index_uni, paper_edge_index_inv], dim=1)
    return paper_edge_index 

def create_paper_edge_index(edge_index, num_papers, num_authors, ratio=0.1):
    print(ratio)
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

    # 0. sampling random
    paper_edge_index = sampling_uniform(paper_edge_index, ratio)
    # 1. sampling with lower prob whose author has high degree
    # paper_edge_index = sampling_inv_to_degree_of_neighbors(edge_index, paper_edge_index, num_papers, num_authors, ratio)
    # 2. sampling mixed 
    # paper_edge_index = sampling_mixed(edge_index, paper_edge_index, num_papers, num_authors, ratio) 

    print(paper_edge_index.shape)

    # input()
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

def emptying_ckpts():
    import glob 
    print('remove all ckpts?')
    input()
    path = 'checkpoints'
    files = glob.glob('checkpoints/*')
    for f in files:
        os.remove(f)

def save_model(model, epoch):
    path = f'checkpoints/model_{epoch}.pth'
    torch.save(model.state_dict(), path)


