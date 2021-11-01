import torch 

def construct_negative_graph(edge_index, num_nodes, k, device): # k: number of negative examples
    src, dst = edge_index[0,:], edge_index[1,:]
    neg_src = src.unsqueeze(1).repeat((1,k))
    neg_dst = torch.randint(0, num_nodes, (len(src)*k,)).unsqueeze(1).to(device)
    neg_edge_index = torch.cat([neg_src, neg_dst], dim=1).T
    return neg_edge_index  