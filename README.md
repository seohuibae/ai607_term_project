# ai607_term_project
## Task  
Same author detection problem 

## Get started 
### required 
- python (>=3.6)
- pytorch (>=1.4) 
- pytorch_geometric (https://github.com/pyg-team/pytorch_geometric) 

### how to install 
```
conda env create --file environment.yaml
```

## Model description
### baseline 
initial paper/author node embedding \
- nn.Embedding whose dimension is the number of paper/author \
node embedding extraction \
- GCN \
link prediction \
- minimizing triplet loss (positive pairs gets closer, negative pairs gets distant)\
- a bilinear projection to measure the distance of two embeddings\

### ours 
\

## How to train
### baseline 
```
python train_baseline.py 
```

