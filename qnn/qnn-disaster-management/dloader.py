import os
import numpy as np
from torch.utils.data import Dataset
from proc import loadImageToTensor, loadGroundTruthImage

class LPCVC23Data(Dataset):
    def __init__(self, set_name, data_root):
        assert set_name in ['train', 'val']
        self.prefix_path_img = os.path.join(data_root, 'data', 'IMG', set_name)
        self.prefix_path_gt = os.path.join(data_root, 'data', 'GT', set_name)
        
        self.data_img = sorted(os.listdir(self.prefix_path_img))
        self.data_gt = sorted(os.listdir(self.prefix_path_gt))
        
    def get_image(self, idx):
        img_path = os.path.join(self.prefix_path_img, self.data_img[idx])
        return loadImageToTensor(img_path)
    
    def get_ground_truth(self, idx):
        gt_path = os.path.join(self.prefix_path_gt, self.data_gt[idx])
        return loadGroundTruthImage(gt_path)
                             
    def __len__(self):
        return len(self.data_img)
    
    def __getitem__(self, idx):
        return {
            'img': self.get_image(idx),
            'gt': self.get_ground_truth(idx),
            'idx': idx
        }

