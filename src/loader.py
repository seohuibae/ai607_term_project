import torch 
import numpy as np 
from torch_geometric.data  import Data
from torch_geometric.data import DataLoader 

from itertools import combinations

class BipartiteData(Data):
    def __init__(self, edge_index, x_s, x_t, N_s, N_t): # s: paper, t: author
        super(BipartiteData, self).__init__() 
        self.edge_index = edge_index 
        self.x_s = x_s 
        self.x_t = x_t
        self.N_s = N_s 
        self.N_t = N_t
    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([self.x_s.size(), self.x_t.size()])
        else:
            return super(BipartiteData, self).__init__(key, value)

class AuthorData(Data):
    def __init__(self, true_edge_index, false_edge_index):
        self.true_edge_index = true_edge_index
        self.false_edge_index = false_edge_index
        

DATA_ROOT_DIR = '../dataset/'

def flatten(t):
    return [item for sublist in t for item in sublist]

# def indices_to_one_hot(data, nb_classes):
#     """Convert an iterable of indices to one-hot encoded labels."""
#     targets = np.array(data).reshape(-1)
#     return np.eye(nb_classes)[targets]

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def load_relation_data(device, fname='paper_author_relationship.csv'):
    train_data = ''
    with open(DATA_ROOT_DIR+fname) as f: 
        content = f.readlines() 
        N_s = len(content)
        paper_ints = [i for i in range(N_s)]
        content = [a.strip().split(',') for a in content]
        author_strs = set(flatten(content))
        N_t = len(author_strs)
        author_ints = [i for i in range(N_t)]
        author_dict = dict(zip(author_strs, author_ints))
    # one-hot encoded initial feature vectors
    # x_s = indices_to_one_hot(paper_ints, N_s) 
    # x_t = indiceS_to_one_hot(author_ints, N_t)
    x_s = torch.tensor(paper_ints) 
    x_t = torch.tensor(author_ints)

    paper_dict = dict()
    for idx, author in zip(paper_ints, content): 
        paper_dict[idx] = author

    # edge index
    edge_index = []
    for p_int in range(len(content)): 
        authors = content[p_int]
        for a_str in authors: 
            a_int = author_dict[a_str]
            edge_index.append([p_int, a_int])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    data = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, N_s=N_s, N_t=N_t)

    return data, author_dict, paper_dict

def load_data(device):

    data, author_dict, paper_dict = load_relation_data(device)

    folds = ['train', 'valid', 'query']

    with open(DATA_ROOT_DIR+'train_dataset.csv') as f: 
        content = f.readlines() 
        content = content[1:]
        content = [[tmp.strip() for tmp in a.strip().split(',')] for a in content]
        true = []
        for a in content: 
            true.append([author_dict[a[0]], author_dict[a[1]]])
        train_true_samples = torch.tensor(true) 

    false = []
    for _, authors in enumerate(paper_dict.items()):
        authors = authors[1]
        for pair in combinations(authors, 2):
            false.append([author_dict[pair[0]], author_dict[pair[0]]])
    train_false_samples = torch.tensor(false)

    with open(DATA_ROOT_DIR+'valid_dataset.csv') as f: 
        content = f.readlines() 
        content = content[1:]
        content = [[tmp.strip() for tmp in a.strip().split(',')] for a in content]
        true = []
        false = []
        for a in content: 
            if a[2] == 'True':
                true.append([author_dict[a[0]], author_dict[a[1]]])
            else:
                false.append([author_dict[a[0]], author_dict[a[1]]])
        valid_true_samples = torch.tensor(true) 
        valid_false_samples = torch.tensor(false)

    with open(DATA_ROOT_DIR+'query_dataset.csv') as f: 
        content = f.readlines() 
        content = content[1:]
        content = [[tmp.strip() for tmp in a.strip().split(',')] for a in content]
        samples = []
        for a in content: 
            samples.append([author_dict[a[0]], author_dict[a[1]]])
        query_samples = torch.tensor(samples) 

    data = data.to(device)
    train_true_samples = train_true_samples.to(device)
    train_false_samples = train_false_samples.to(device)
    valid_true_samples = valid_true_samples.to(device)
    valid_false_samples = valid_false_samples.to(device)
    query_samples = query_samples.to(device)

    return data, train_true_samples, train_false_samples, valid_true_samples, valid_false_samples, query_samples
    
  