import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dloader import LPCVC23Data
from viz import viz_segmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = os.path.abspath('.')
results_dir = os.path.abspath('results')
batch_size = 1
num_workers = 3

dl = DataLoader(
    LPCVC23Data('val', data_root),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

for batch_idx, batch in enumerate(dl):
    gt = batch['gt'].squeeze(0)
    idx = batch['idx'].item()
    gt_viz_dir = os.path.join(results_dir,'gt_viz')
    out_img_path = os.path.join(
        gt_viz_dir,
        'gtviz_%02d'%idx +'.png'
    )
    fig = viz_segmentation(gt)
    fig.savefig(out_img_path)
    plt.close(fig)
