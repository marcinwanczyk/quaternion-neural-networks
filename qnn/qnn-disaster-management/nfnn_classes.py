'''nfnn no fat nn

AV/AGH/2023
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from dloader import LPCVC23Data

def make_hist():
    data_root = os.path.abspath('.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pic = 1021
    #n_pic =1
    batch_size = n_pic
    num_workers = 4 # ncpus -1
    train_dataloader = DataLoader(
        Subset(LPCVC23Data('train', data_root), range(n_pic)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            gt = batch['gt']
            h = gt.float().histogram(bins=14,range=(0,13))
            #plt.figure()
            #plt.bar(range(14), h.hist)
            torch.save(h,'histogram_train_gt')
            
def weights_from_hist_file(fname):
    hist_train = torch.load(fname).hist
    hts = hist_train.sum()
    w = hts/hist_train
    return (w/w.max())
        
        
        

        
