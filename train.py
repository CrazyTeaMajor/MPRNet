import torch
import os
import warnings
import math
import torchvision
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
# from model_p2p import myModel, matcher, set_criterion
from model_jcy2 import myModel, matcher, set_criterion
from dataset_p2p import Dataset, my_collate_fn

    
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

shha_root = './SHHA/part_A/train/images'
test_shha_root = './SHHA/part_A/test/images'

jhu_root = './jhu_crowd_v2.0/train/images'
test_jhu_root = './jhu_crowd_v2.0/test/images'


dataset = Dataset(shha_root) 
test_dataset = Dataset(test_shha_root, False) 

# dataset = Dataset(jhu_root) 
# test_dataset = Dataset(test_jhu_root, False) 

epochs = 1000
start_epoch = 0 
load_model = True
load_model = False
batch_size = 8
lr = 1e-05
lr_exp = 10
step_size = 100
gamma = 0.9
empty_weight = 0.5
seed = 42

weight_class = 1
weight_point = 0.025

weight_loss_point = 0.0002

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

matcher = matcher(weight_class, weight_point).to(device)
model = myModel(matcher).to(device)
criterion = set_criterion(1, empty_weight, batch_size, device).to(device)
criterion_mse = nn.MSELoss(reduction = 'mean').to(device)
criterion_mae = nn.L1Loss(reduction = 'mean').to(device)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, collate_fn=my_collate_fn, drop_last=True,shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8, collate_fn=my_collate_fn, drop_last=True,shuffle=False)

bb_param = list(map(id, model.body1.parameters())) + \
            list(map(id, model.body2.parameters())) + \
            list(map(id, model.body3.parameters())) + \
            list(map(id, model.body4.parameters()))



bb_params = filter(lambda p: id(p)  in bb_param, model.parameters())
neck_params = filter(lambda p: id(p)  not in bb_param, model.parameters())


optimizer = torch.optim.Adam([
            {'params': bb_params, 'lr': lr},
            {'params': neck_params, 'lr': lr * lr_exp}])


if load_model:
    print('load_model')
    # checkpoint = torch.load("./checkpoint/epoch_187.pth")
    checkpoint = torch.load("./checkpoint/epoch_best.pth")
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print('init_model')


scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
# model = torch.compile(model)

min_mae = torch.tensor(1000.11)
for e in range(start_epoch + 1, epochs):
    Loss_ce = 0
    Loss_point = 0
    Loss_mae = 0
    Loss_mse = 0
    model.train()
    for i,(img,label) in enumerate(data_loader):
        img = img.to(device)
        gt_num = []
        pd_num = []
        for j in range(len(label)):
            label[j]['point'] = label[j]['point'].to(device)
            gt_num.append(len(label[j]['point']))
            
        feature_map, pre_points, indices = model(img, label)
        
            
        
        loss_ce, loss_point = criterion(feature_map, pre_points, indices, label)
        
        
        Loss_ce += loss_ce
        Loss_point += loss_point

        
        optimizer.zero_grad()
        loss_ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    
    # break
    
    scheduler.step()

    
    
    model.eval()
    MAE = 0
    MSE= 0
    gap = 8
    for i,(img, label) in enumerate(test_data_loader):
        # print(img.shape)
        img = img.to(device)
        feature_map, c2 = model(img)
        
#         height = img.shape[2]
#         width = img.shape[3]
#         f_map = feature_map[0].softmax(-1) # f_map.shape = [num_queries, 2]
#         f_map2 = f_map[:,1].reshape(height // 4, width // 4)
#         b = c2[0].softmax(-1)
#         b = b[:,1].reshape(height // 4 // gap, width // 4 // gap)

#         for i in range(b.shape[0]):
#             for j in range(b.shape[1]):
#                 idx = f_map2[i*gap:i*gap+gap, j*gap:j*gap+gap].argmax()
#                 x = idx // gap
#                 y = idx - gap * x
#                 f_map2[i*gap+x,j*gap+y] = max(f_map2[i*gap+x,j*gap+y], b[i,j])

#         f_map2 = f_map2.reshape(height * width // 16, 1)
#         f_map[:,1] = f_map2.view(-1)
        
#         prob = f_map[:,1]

        prob = feature_map.softmax(-1)[0][:,1]
      
        
        pred_num = torch.sum(prob > 0.5)
        
        gt_num = len(label[0]['point'])
        
        MAE += abs(gt_num - pred_num)
        MSE += math.pow(gt_num - pred_num, 2)
    
    
    state = {
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    
    if MAE / len(test_data_loader) < min_mae:
        min_mae = MAE/ len(test_data_loader)
        torch.save(state, "./checkpoint/epoch_" + str(e) + ".pth")
        torch.save(state, "./checkpoint/epoch_best.pth")
    
    print('epoch:', e, 
          # ' lr:', optimizer.param_groups[0]['lr'],
          ' loss_ce:', np.round(Loss_ce.detach().cpu().numpy() / len(data_loader), 4), 
          ' loss_point:', np.round(Loss_point.detach().cpu().numpy() / len(data_loader), 3), 
          # ' loss_mae:', np.round(Loss_mae.detach().cpu().numpy() / len(data_loader), 3), 
          # ' loss_mse:', np.round(Loss_mse.detach().cpu().numpy() / len(data_loader), 3), 
          " MAE:", np.round(MAE.detach().cpu().numpy() / len(test_data_loader), 3),
          " MSE:", np.round(math.pow(MSE / len(test_data_loader), 0.5), 3),
          ' min_MAE:', np.round(min_mae.detach().cpu().numpy(), 3))
    
    with open("./record/loss_jhu_jcy2_fuse.txt", 'a') as file:
           file.write("Epoch: " + str(e) + 
                      " MAE: " + str(MAE.detach().cpu().numpy()/len(test_data_loader)) + 
                      " MSE: " + str(np.round(math.pow(MSE/len(test_data_loader), 0.5), 3)) + 
                      " best_MAE: " + str(min_mae.detach().cpu().numpy()) + "\n")
        
        
            
            
   

        
    