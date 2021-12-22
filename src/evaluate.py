import torch 
import numpy as np
import itertools 

def predict_query_answer(query_sample, pos_score, neg_score, thr, reverse_dict):
    prediction = torch.ones((len(pos_score),))*(-1)
    indices = torch.tensor([i for i in range(len(pos_score))])

    pos_mask_pos = (pos_score>=thr).squeeze(1).to(torch.bool) 
    pos_mask_neg = (neg_score<thr).squeeze(1).to(torch.bool)
    pos_mask = pos_mask_pos
    # pos_mask = pos_mask_pos * pos_mask_neg

    neg_mask_pos = (pos_score>=thr).squeeze(1).to(torch.bool) 
    neg_mask_neg = (neg_score<thr).squeeze(1).to(torch.bool)
    neg_mask = neg_mask_neg
    # neg_mask = neg_mask_pos * neg_mask_neg

    pos = indices[pos_mask]
    neg = indices[neg_mask]
    # print('validity check')
    # print('num pos: ', len(pos))
    # print('num neg: ', len(neg))
    
    prediction[neg] = 0
    prediction[pos] = 1 
    
    answer = torch.cat([query_sample.cpu().detach(), prediction.unsqueeze(1)], dim=1)
    answer = answer.numpy().astype(int)

    path = 'query_answer.csv'
    f = open(path, 'w')
    f.write("ID, ID, label \n")
    for ans in answer:
        ans_str=[]
        ans_str.append(str(reverse_dict[ans[0]]))
        ans_str.append(str(reverse_dict[ans[1]]))
        if ans[2]==1:
            ans_str.append("True")
        else:
            ans_str.append("False")
        out = ', '.join(ans_str) 
        f.write(out+'\n')    
    f.close()
    
def is_pair_in_list_of_pairs(pair, list_of_pairs):
    isin = False
    for p in list_of_pairs: 
        if pair[0]==p[0] and pair[1]==p[1]:
            isin=True 
        elif pair[0]==p[1] and pair[1]==p[0]:
            isin=True 
    return isin 

def generate_author_pairs(num_authors):

    author_ids = [i for i in range(num_authors)]
    candidate_pool = list(itertools.combinations(author_ids, 2))
    candidate_pool = torch.tensor(np.array(candidate_pool))
    perm = torch.randperm(candidate_pool.size(0))
    candidate_pool = candidate_pool[perm]
    return candidate_pool

def generate_random_author_pairs(num_authors,  num_samples):

    candidate_pool = np.random.choice(num_authors, 2*num_samples, replace=False)
    candidate_pool = torch.tensor(candidate_pool)
    candidate_pool = candidate_pool.reshape(num_samples,2)
    
    # print(candidate_pool.shape)
    return candidate_pool

def save_same_author(same_authors, reverse_dict): 
    path = 'same_author.csv'
    f = open(path, 'w')
    f.write("ID, ID \n")
    for ans in same_authors:
        ans_str = [str(reverse_dict[a]) for a in ans]
        out = ', '.join(ans_str) 
        f.write(out+'\n')    
    f.close()

def remove_present_author(given_samples, same_authors):
    given_samples= set(given_samples)
    same_authors = set(same_authors)
    remaining = list(same_authors - given_samples) 
    return list(remaining)

def predict_same_author( candidate_pool, pos_score, neg_score, thr):
    prediction = torch.zeros((len(pos_score),))
    indices = torch.tensor([i for i in range(len(pos_score))])
    pos = indices[(pos_score>=thr).squeeze(1).to(torch.bool)]
    neg = indices[(neg_score<thr).squeeze(1).to(torch.bool)]
    prediction[pos] = 1 
    prediction[neg] = 0

    same_authors = []
    cnt_pos = 0
    
    candidate_pool = candidate_pool.cpu().tolist()

    for i in pos: 
        pos_pair = candidate_pool[i]
        same_authors.append((pos_pair[0], pos_pair[1]))
        cnt_pos +=1 
    
    return same_authors

def predict(pos_score, neg_score):
    thr = torch.mean(torch.cat([pos_score, neg_score], dim=0))

    pos_pred = torch.zeros((len(pos_score),))
    neg_pred = torch.ones((len(neg_score),))

    pos_indices = torch.tensor([i for i in range(len(pos_score))])
    neg_indices = torch.tensor([i for i in range(len(neg_score))])

    pos_correct = pos_indices[(pos_score>=thr).squeeze(1).to(torch.bool)]
    neg_correct = neg_indices[(neg_score<thr).squeeze(1).to(torch.bool)]

    pos_pred[pos_correct] = 1
    neg_pred[neg_correct] = 0

    return pos_pred, neg_pred, thr 

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

# def predict_same_author(given_samples, candidate_pool, score, thr):
#     path = 'same_author.csv'
#     f = open(path, 'w')
#     f.write("ID, ID ")

#     prediction = torch.zeros((len(score),))
#     indices = torch.tensor([i for i in range(len(score))])
#     pos = indices[(score>=thr).squeeze(1).to(torch.bool)]
#     neg = indices[(score<thr).squeeze(1).to(torch.bool)]
#     prediction[pos] = 1 
#     prediction[neg] = 0

#     same_authors = []
#     cnt_pos = 0
    
#     print(len(pos))
#     for i in pos: 
#         if cnt_pos>1000:
#             break 
#         pos_pair = candidate_pool[i]
#         print(pos_pair)
#         if not is_pair_in_list_of_pairs(pos_pair, given_samples): 
#             pos_pair = pos_pair.cpu().tolist()
#             same_authors.append(pos_pair)
#             cnt_pos +=1 
#             out = ', '.join(pos_pair) 
#             f.write(out+'\n') 
#             print(cnt_pos)
#     f.close()
#     print(cnt_pos)

    

# def predict_same_author(given_samples, num_authors, total_num=1000):
#     import itertools 
#     path = 'same_author.csv' 
#     cnt = 0
#     k = 10 
#     k = 15

#     author_ids = [i for i in range(num_authors)]
#     candidate_pool = torch.tensor(np.array(itertools.product(author_ids, author_ids)))
#     perm = torch.randperm(candidate_pool.size(0))
#     # idx = perm[:k]
#     candidate_pool = candidate_pool[perm]
    
#     f = open(path, 'w')
#     f.write("ID, ID, ")
#     while cnt <= total_num:
#         # candidate = np.random.choice(num_authors, 2, replace=False) # sampling 
#         candidate = 
#         if is_pair_in_list_of_pairs(candidate, given_samples):
#             pass
        
#         cnt+=1


