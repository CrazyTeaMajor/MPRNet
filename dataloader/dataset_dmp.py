import os
import cv2
import torch
import math
import random
import numpy as np
import torch.nn as nn
import matplotlib.image as ig
import scipy.io as scio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class Dataset(Dataset):
    def __init__(self, img_root, jhu_img_root, train = True):
        self.img_path_list = []
        self.dmap_list = []
        self.gt_number = []
        self.train = train
        name_list = os.listdir(img_root)
        
        for i in range(len(name_list)):
            if name_list[i].endswith(".jpg"):
                img_path = os.path.join(img_root, name_list[i])
                self.img_path_list.append(img_path)
                if self.train:
                    dmap_path = img_path.replace("images", "gaus_maps").replace(".jpg", ".npy")
                    self.dmap_list.append(dmap_path)
                else:
                    gt_path = img_path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
                    gt = scio.loadmat(gt_path)
                    gt_map = gt["image_info"][0][0][0][0][0]
                    self.gt_number.append(len(gt_map))
        
#         jhu_gaus_root = jhu_img_root.replace('images', 'gaus_maps').replace('.jpg','.npy')
#         name_list2 = os.listdir(jhu_gaus_root)
        
#         for i in range(len(name_list2)):
#             if name_list2[i].endswith(".npy"):
#                 img_path = os.path.join(jhu_img_root, name_list2[i]).replace('.npy', '.jpg')
#                 self.img_path_list.append(img_path)
#                 dmap_path = img_path.replace("images", "gaus_maps").replace(".jpg", ".npy")
#                 self.dmap_list.append(dmap_path)
        
        self.nSamples = len(self.img_path_list)
        
        
        
    def __len__(self):

        return self.nSamples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img = Image.open(self.img_path_list[index]).convert('RGB')
        
        scale = 1
        if img.size[0] * img.size[1] > 3000 * 3000:
            scale = math.pow(img.size[0] * img.size[1] / 3000 / 3000, 0.5)
            
        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                       ])
        
        if self.train:
            dmap = np.load(self.dmap_list[index])
            dmap = cv2.resize(dmap,(int(dmap.shape[1] / 8), int(dmap.shape[0] / 8)),interpolation = cv2.INTER_CUBIC)*64

            random_num = random.randint(0, 100)
            
            # if random_num > 50:
            #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #     dmap = np.fliplr(dmap)


            dmap = torch.FloatTensor(dmap.copy()).unsqueeze(0)

            img = transform(img).float()
            # img = torch.FloatTensor(np.array(img)).permute(2,0,1) / 255

            img, dmap = self.crop(img, dmap, 384, 512)

            return img, dmap
        
        else:
            # new_size = (int(img.size[0] / scale) // 32 * 32, int(img.size[1] / scale) // 32 * 32)
            # img = img.resize(new_size)
            
            img = transform(img).float()
            
            new_h = img.shape[1]
            new_w = img.shape[2]
            if img.shape[1] % 32 != 0:
                new_h = (img.shape[1] // 32 + 1) * 32
                
            if img.shape[2] % 32 != 0:
                new_w = (img.shape[2] // 32 + 1) * 32
                
            new_img = torch.zeros((3, new_h, new_w))
            new_img[:, :img.shape[1], :img.shape[2]] = img
            img = new_img
           
            # img = torch.FloatTensor(np.array(img)).permute(2,0,1) / 255
            
            return img, self.gt_number[index]
    
    
    
    def crop(self, img, dmap, height, width):

        o_height = img.shape[1]
        o_width = img.shape[2]

        if o_height <= height or o_width <= width:
            img = torch.randn(3,height,width)
            dmap = torch.FloatTensor(torch.zeros(1, height // 8, width // 8))
        else:
            delta_h = o_height - height
            delta_w = o_width - width

            k_h = random.randrange(0, delta_h, 8)
            k_w = random.randrange(0, delta_w, 8)

            img = img[:,k_h:k_h+height,k_w:k_w+width]
            dmap = dmap[:,k_h//8:(k_h+height)//8,k_w//8:(k_w+width)//8]

        return img, dmap

    
        
