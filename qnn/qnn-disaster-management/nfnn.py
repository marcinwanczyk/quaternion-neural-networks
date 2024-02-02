'''Quaternion/Real valued Neural Network script
MW/AGH/2023
AV/AGH/2023
'''

import os
import time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from models.fanet import FANet as FANetR
from models.qnet import FANet as FANetQ
from dloader import LPCVC23Data
from proc import (
    scale_prediction,
    prediction_to_classes,
    prediction_to_np_img
)

from viz import save_viz_segmentation
from nfnn_classes import weights_from_hist_file

# torch.set_num_threads(1)
model_name = 'FANetR'
# model_name = 'FANetQ'
# model_to_load = 'Model-lpcvcR_epoch2'
model_to_load = None
train_data = 1021
#train_data = 2
num_epochs = 10
num_workers = 15 # cpus-1
# learning_rate = 1e-4
learning_rate = 0.0032
weight_decay = 1e-7
batch_size = 32
momentum = 0.99
label_smoothing_regularization = 0.01
hist_file_name = 'histogram_train_gt'

results_dir = os.path.abspath('results-FANetR')
data_root = os.path.abspath('.')

models = {'FANetQ':FANetQ, 'FANetR':FANetR}
params_dir = os.path.join(results_dir,'params')
img_out_dir = os.path.join(results_dir,'img_out')
descriptor = 'Model-' + model_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models[model_name]().to(device)
model.name = model_name

#if model_to_load is not None:
 #   checkpoint = torch.load(os.path.join(params_dir, model_to_load))
 #   model.load_state_dict(checkpoint['params'])

nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Model: {}.'.format(model.name), 'Number of trainable params: ', nb_params)

hist_file = os.path.join(results_dir,hist_file_name)
w = weights_from_hist_file(hist_file).to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay
)

# 1021
train_dataloader = DataLoader(
    Subset(LPCVC23Data('train', data_root), range(train_data)),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
# 100
test_dataloader = DataLoader(
    LPCVC23Data('val', data_root),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)

# loss_func = nn.MSELoss()
loss_func = nn.CrossEntropyLoss(
    weight=w,
    label_smoothing=label_smoothing_regularization
)

def main():    
    def train(dataloader, model, loss_func, optimizer):
        train_size = len(train_dataloader.dataset)
        model.train()
        epoch_loss = 0    
        for batch_idx, batch in enumerate(dataloader):
            img = batch['img'].to(device)
            gt = batch['gt'].to(device).long()
            optimizer.zero_grad()
            out = model(img)        
            out = scale_prediction(out)
            out = out.requires_grad_()
            # N batch size, C class size, W width pix, H height pix
            # out(loss input, logits, float, requires_grad =true) NxCxWxH  Nx14x512x512  
            # gt (loss target, long) NxWxH Nx512x512
            # print(out.size(), out.type())
            loss = loss_func(out, gt)

            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
            if (batch_idx% 100) == 0:
                loss, current = loss.item(), (batch_idx +1)
                print(f"loss: {loss:>7f} [{current:>5d}/{train_size:>5d}]")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_loss = epoch_loss/train_size
        print('epoch_loss train avg: {loss:>7f}'.format(loss=epoch_loss))
        return epoch_loss


    def test(dataloader, model, loss_func, epoch):
        test_size = len(dataloader.dataset)
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                img = batch['img'].to(device)
                gt = batch['gt'].to(device).long()
                with torch.set_grad_enabled(False):
                    out = model(img)
                    out = scale_prediction(out) 
                    loss = loss_func(out, gt)                
                    out = prediction_to_classes(out)          
                for r in range(batch['idx'].numel()):
                    idx = batch['idx'][r].item()
                    out_img_path = os.path.join(
                        img_out_dir,
                        'img_%02d'%idx + '_e_%02d'%epoch + '.png'
                    )

                    out_img = prediction_to_np_img(out[r])
                    save_viz_segmentation(out_img, out_img_path)

                epoch_loss = epoch_loss + loss.item()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        epoch_loss = epoch_loss/test_size
        print('epoch_loss test avg: {loss:>7f}'.format(loss=epoch_loss))    
        return epoch_loss


    train_loss = np.zeros((num_epochs))
    test_loss = np.zeros((num_epochs))
    for epoch in range(num_epochs):
        print('Epoch', epoch)   
        start_time = time.time() 
        train_loss[epoch] = train(train_dataloader, model, loss_func, optimizer)
        test_loss[epoch] = test(test_dataloader, model, loss_func, epoch)

        check_point = {'params': model.state_dict(),
                       'optimizer': optimizer.state_dict()}
        torch.save(check_point,
                   os.path.join(params_dir, descriptor+'_epoch'+str(epoch)))
        end_time = time.time()
        print('Elapsed time: ', end_time - start_time)
    import matplotlib.pyplot as plt    
    plt.figure()
    plt.title('Model: {}'.format(model.name))
    plt.plot(train_loss, 'o-', label='train_loss')
    plt.plot(test_loss, 'o-', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(params_dir,'train-test.png'))

if __name__ == "__main__":
    main()
