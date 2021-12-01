import argparse
import numpy as np
import os
import random
import torch 


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--gpus', type=str, default='0',help='device to use')  #
    parser.add_argument('--setting', type=str, default="description of hyper-parameters.")  #
    parser.add_argument('--early_stop', type=int, default= 20, help='early_stop')
    parser.add_argument('--seed',type=int, default=1234, help='seed')


    # shared parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay',type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--hiddens', type=str, default='64')
    parser.add_argument('--pred_emb_dim', type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--act', type=str, default='leaky_relu', help='activation funciton')  #
    parser.add_argument('--normalize', action='store_true') # mnormalize feature 
    parser.add_argument('--l2reg', action='store_true')

    parser.add_argument('--expname', type=str, default='')
    args, _ = parser.parse_known_args()

    return args

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def set_seed(seed, use_cuda=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True 

def set_device():
    # set gpu id and device 
    use_cuda = torch.cuda.is_available() 
    if len(args.gpus) > 1:
        print('multi gpu')
        # raise NotImplementedError 
        device = torch.device("cuda")
    else:  
        gpu_id = args.gpus[0]
        device = torch.device("cuda:"+str(gpu_id))
        torch.cuda.set_device(gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return device 

args = get_params()
params = vars(args)

args.gpus = parse_gpus(args.gpus)
SVD_PI = True

set_seed(args.seed, True)
device = set_device()



