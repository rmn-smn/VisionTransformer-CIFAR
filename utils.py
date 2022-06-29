import os
import torch
import random
import numpy as np

def seed_everything(seed=1234):
    torch.use_deterministic_algorithms(True) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True