from collections import Counter
from utils import *
import numpy as np


def main():
    edge_index, true_samples, false_samples, pos_pred, neg_pred = load_pred_data()

    sample_idx = np.array([i for i in range(len(pos_pred))])
    # print(pos_pred==0)
    tps_idx = sample_idx[(pos_pred==1)]
    fns_idx = sample_idx[(pos_pred==0)]
    fps_idx = sample_idx[(neg_pred==1)]
    tns_idx = sample_idx[(neg_pred==0)]

    # print(FNs_idx)
    tp_authors = true_samples[:, tps_idx]
    fn_authors = true_samples[:,fns_idx]
    fp_authors = false_samples[:,fps_idx]
    tn_authors = false_samples[:, tns_idx]
    print(len(fn_authors))
    print(len(fp_authors))
    print(fn_authors.shape)
    print(fp_authors.shape)
    input()
    degree = degree_dict_authors(edge_index)
    degree_papers = degree_dict_papers(edge_index)
    deg_cnt = Counter(list(degree.values()))
    mean_degree = sum(degree.values())/len(degree)
    mean_degree_paper = sum(degree_papers.values())/len(degree_papers)

    # print(deg_cnt)
    print(mean_degree)
    print(mean_degree_paper)
    print(max(degree_papers.values()), min(degree_papers.values()))


    input()

    paper_edge_index = edge_index[0]
    author_edge_index = edge_index[1]
    i=0
    while True:
        author = tp_authors[:, i].tolist()
        data = [degree[author[0]],degree[author[1]]]
        print(data)
        author = tn_authors[:, i].tolist()
        data = [degree[author[0]],degree[author[1]]]
        print(data)

        author = fn_authors[:, i].tolist()
        data = [degree[author[0]],degree[author[1]]]
        print(data)
        #author = fp_authors[:, i].tolist()
        #data = [degree[author[0]],degree[author[1]]]
        #print(data)

        tmp = np.concatenate([paper_edge_index[author_edge_index==author[0]], paper_edge_index[author_edge_index==author[1]]])
        deg_tmp = []
        for p in tmp: 
            deg_tmp.append(degree_papers[p])
        print(deg_tmp)
        print(sum(deg_tmp)/len(deg_tmp))


        i+=1
        input()
    # fn_authors_degree = []
    # for i in range(len(fn_authors[0])):
    #     author = fn_authors[:,i].tolist()
    #     # print(author)
    #     data = [degree[author[0]], degree[author[1]]]
    #     fn_authors_degree.append(data)
    #     print(data)
    #     input()
    # # print(fn_authors_degree)
    # fp_authors_degree = []
    # for i in range(len(fp_authors[0])):
    #     author = fp_authors[:,i].tolist()
    #     data = [degree[author[0]], degree[author[1]]]
    #     fp_authors_degree.append(data)
    #     print(data)
    #     input()


if __name__ == "__main__":
    main()
