import os
import random
import warnings
import torch
import numpy as np


def set_seed(SEED):
    if SEED:
        print('RANDOM SEED:', SEED)
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False 

def setup(args):
    warnings.filterwarnings("ignore")
    os.environ['TORCH_HOME'] = args.torch_home
    set_seed(args.seed)
