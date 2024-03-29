import os
import math
import cv2
import torch
import numpy as np
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import matplotlib.image as ig
from tqdm import tqdm
from PIL import Image
from model_jcy import myModel
from model_jcy2 import myModel as jcy2
# from model_dmp import myModel
from torchvision import datasets, transforms


def getDmap(img_root, epoch, index):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    model = myModel().to(device)
    checkpoint = torch.load("./checkpoint_dmp/epoch_" + str(epoch) + ".pth")
    model.load_state_dict(checkpoint['state_dict'])
    
    img_path = img_root + str(index) + ".jpg"
    gaus_path = img_path.replace("images", "gaus_maps").replace(".jpg",".npy")
    gaus_map = np.load(gaus_path)
   
        
    img = Image.open(img_path).convert('RGB')
    new_size = (img.size[0] // 32 * 32, img.size[1] // 32 * 32)
    img = img.resize(new_size)
        

    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
               ])

    img = transform(img).unsqueeze(0).float().to(device)
    dmap = model(img).detach().cpu().numpy().squeeze(0).squeeze(0)
    
    
    return gaus_map, dmap
        


@torch.no_grad()
def calculate_MAE_sha_dmap(img_root, epoch):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    model = myModel().to(device)
    checkpoint = torch.load("./checkpoint_dmp/epoch_" + str(epoch) + ".pth")
    model.load_state_dict(checkpoint['state_dict'])
    
    MAE = 0
    model.eval()
    for index in range(1, 183):
        # print(index)
        img_path = img_root + '/IMG_' + str(index) + ".jpg"
        gt_path = img_path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
        gt = scio.loadmat(gt_path)
        gt_map = gt["image_info"][0][0][0][0][0]
        
        img = Image.open(img_path).convert('RGB')
        
#         new_size = (img.size[0] // 32 * 32, img.size[1] // 32 * 32)
#         img = img.resize(new_size)
        
        gt_num = len(gt_map)

        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                   ])
        
        
        img = transform(img).float()
        new_h = img.shape[1]
        new_w = img.shape[2]
        if img.shape[1] % 32 != 0:
            new_h = (img.shape[1] // 32 + 1) * 32

        if img.shape[2] % 32 != 0:
            new_w = (img.shape[2] // 32 + 1) * 32

        new_img = torch.zeros((3, new_h, new_w))
        new_img[:, :img.shape[1], :img.shape[2]] = img
        img = new_img.unsqueeze(0).to(device)
        
        
        output = model(img)
        
        dmap = output.squeeze(0).squeeze(0).detach().cpu().numpy()
        
        pred_num = output.sum()
        MAE += abs(gt_num - pred_num)
        
        
        # outfile = img_path.replace('images', 'pred_gaus_maps').replace('.jpg','')
        # np.save(outfile, dmap)
        # break
        
    print("{} MAE_Loss: {}".format(epoch, MAE / 182))
    
    
    
    # with open("./record_dmap/test_loss_shha.txt", 'a') as file:
    #         file.write("Epoch: " + str(epoch) + 
    #                   " MAE: " + str(MAE.detach().cpu().numpy()/182) + "\n")
    
def check_p2p(img_path, epoch, scale, thresh=0.5, markersize=5):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = myModel().to(device)
    # checkpoint = torch.load("./checkpoint/epoch_" + str(epoch) + ".pth")
    checkpoint = torch.load("./checkpoint/epoch_jcy_54.43.pth")
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    gt_num = 0
    gt_map = []
    path = img_path
    if path.find('SHHA') == -1:
        gt_path = path.replace('images', 'gt').replace('.jpg', '.txt')

        with open(gt_path, 'r') as file:
            lines = file.readlines()
            gt_num = len(lines)
            for line in lines:
                data = line.split()
                x = float(data[0])
                y = float(data[1])
                gt_map.append([x, y])
    else:
        gt_path = path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
        gt = scio.loadmat(gt_path)
        gt_map = gt["image_info"][0][0][0][0][0]
        gt_num = len(gt_map)
    
    # gt_path = img_path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
    # gt = scio.loadmat(gt_path)
    # gt_map = gt["image_info"][0][0][0][0][0]

    img = Image.open(img_path).convert('RGB')
    
    
    img_draw = img.copy()
    width = img_draw.size[0]
    height = img_draw.size[1]

    gt_num = len(gt_map)

    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
               ])

    img = transform(img)
        
    new_h = img.shape[1]
    new_w = img.shape[2]
    if img.shape[1] % 32 != 0:
        new_h = (img.shape[1] // 32 + 1) * 32

    if img.shape[2] % 32 != 0:
        new_w = (img.shape[2] // 32 + 1) * 32

    new_img = torch.zeros((3, new_h, new_w))
    new_img[:, :img.shape[1], :img.shape[2]] = img
    img = new_img.unsqueeze(0).to(device)

    output = model(img)
    
    prob = output.softmax(-1)[0][:,1]
    pred_num = torch.sum(prob > thresh)
    e_rate = abs(gt_num - pred_num) * 100 / gt_num
    
    print('gt_num: {} pred_num: {} e_rate: {:.3f}%'.format(gt_num, pred_num, e_rate))
    
    X, Y = [], []
    
    dmap = prob.reshape(new_h // 4, new_w // 4)
    
    for i in range(dmap.shape[0]):
        for j in range(dmap.shape[1]):
            if dmap[i][j] > thresh:
                Y.append(i * 4)
                X.append(j * 4)
    
    figsize = (9,9)
    plt.figure(1)
    plt.figure(figsize=figsize)
    plt.imshow(img_draw)
    for i in range(len(X)):
        plt.plot(X[i], Y[i], '.', color='lime',markersize=markersize)
        
    plt.figure(2)
    plt.figure(figsize=figsize)
    plt.imshow(img_draw)
    # for i in range(len(gt_map)):
    #     plt.plot(gt_map[i][0], gt_map[i][1], '.',color='lime', markersize=markersize)
    plt.show()
    
def check_SHHA_dmap(img_path, epoch, scale):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = myModel().to(device)
    checkpoint = torch.load("./checkpoint_dmp/epoch_" + str(epoch) + ".pth")
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    gt_path = img_path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
    gt = scio.loadmat(gt_path)
    gt_map = gt["image_info"][0][0][0][0][0]
    
    gaus_path = img_path.replace('images', 'gaus_maps').replace('.jpg','.npy')
    gt_dmap = np.load(gaus_path)

    img = Image.open(img_path).convert('RGB')
    
    
    img_draw = img.copy()
    width = img_draw.size[0]
    height = img_draw.size[1]

    gt_num = len(gt_map)

    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
               ])

    img = transform(img)
        
    new_h = img.shape[1]
    new_w = img.shape[2]
    if img.shape[1] % 32 != 0:
        new_h = (img.shape[1] // 32 + 1) * 32

    if img.shape[2] % 32 != 0:
        new_w = (img.shape[2] // 32 + 1) * 32

    new_img = torch.zeros((3, new_h, new_w))
    new_img[:, :img.shape[1], :img.shape[2]] = img
    img = new_img.unsqueeze(0).to(device)

    output = model(img)
    
    pred_num = output.sum().detach().cpu().numpy()
    dmap = output.squeeze(0).squeeze(0).detach().cpu().numpy()
    dmap = cv2.resize(dmap,(int(dmap.shape[1] * 8), int(dmap.shape[0] * 8)),interpolation = cv2.INTER_CUBIC)/64
    
    print('gt_num:{} pred_num:{}'.format(gt_num, pred_num))
    
    figsize = 9
    plt.figure(1)
    plt.figure(figsize=(figsize,figsize))
    plt.imshow(img_draw)
    
    plt.figure(2)
    plt.figure(figsize=(figsize,figsize))
    plt.imshow(gt_dmap, cmap=CM.jet)
    plt.show()
    
    plt.figure(3)
    plt.figure(figsize=(figsize,figsize))
    plt.imshow(dmap[:height, :width], cmap=CM.jet)
    plt.show()
    
    

@torch.no_grad()
def calculate_MAE_p2p(img_root, epoch):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    model = jcy2().to(device)
    # checkpoint = torch.load("./checkpoint/epoch_" + str(epoch) + ".pth")
    checkpoint = torch.load("./checkpoint/epoch_174_57.26.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    img_names = os.listdir(img_root)
    path_list = []
    for i in range(len(img_names)):
        if img_names[i].endswith('.jpg'):
            path_list.append(os.path.join(img_root, img_names[i]))
    
    number = len(path_list)
    MAE = 0
    MSE = 0
    E_rate = 0
    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
               ])
    for i, path in enumerate(tqdm(path_list)):
        img = Image.open(path).convert('RGB')
        scale = 1
        if img.size[0] * img.size[1] > 3000 * 3000:
            scale = math.sqrt(img.size[0] * img.size[1] / 3000 / 3000)
            width = int(img.width / scale)
            height = int(img.height / scale)
            img = img.resize((width, height),Image.ANTIALIAS)

            
        # new_size = (int(img.size[0] / scale) // 32 * 32, int(img.size[1] / scale) // 32 * 32)
        # img = img.resize(new_size)
        
        gt_num = 0
        
        if path.find('SHHA') == -1:
            gt_path = path.replace('images', 'gt').replace('.jpg', '.txt')

            with open(gt_path, 'r') as file:
                lines = file.readlines()
                gt_num = len(lines)
        else:
            gt_path = path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
            gt = scio.loadmat(gt_path)
            gt_map = gt["image_info"][0][0][0][0][0]
            gt_num = len(gt_map)

        img = transform(img)
        
        new_h = img.shape[1]
        new_w = img.shape[2]
        if img.shape[1] % 32 != 0:
            new_h = (img.shape[1] // 32 + 1) * 32

        if img.shape[2] % 32 != 0:
            new_w = (img.shape[2] // 32 + 1) * 32

        new_img = torch.zeros((3, new_h, new_w))
        new_img[:, :img.shape[1], :img.shape[2]] = img
        img = new_img.unsqueeze(0).to(device)

        output, c2 = model(img)
        
        prob = output.softmax(-1)[0][:,1]
        pred_num = torch.sum(prob > 0.5)
        
        MAE += abs(gt_num - pred_num)
        MSE += math.pow(gt_num - pred_num, 2)
        if gt_num > 0:
            e_rate = abs(gt_num - pred_num) / gt_num
        else:
            e_rate = 0
            
        E_rate += e_rate
        
        # if e_rate > 0.2:
        #     print('{}\t{}\t{}\t{:.2f}'.format(path, gt_num, pred_num, e_rate))

    print("{} MAE: {} MSE: {} E_rate: {}".format(epoch, MAE / number, math.pow(MSE / number, 0.5), E_rate / number))
    with open("./record/test_loss.txt", 'a') as file:
            file.write("Epoch: " + str(epoch) + 
                      " MAE: " + str(MAE.detach().cpu().numpy()/number) + "\n")
                
