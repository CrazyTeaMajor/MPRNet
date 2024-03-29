import torch
import torch.nn as nn
from torchvision import models
from scipy.optimize import linear_sum_assignment

# f_map.shape = [batch, 2, height, weight]
# loc.shape = [all_points, 3] (x, y, batch_index)
@torch.no_grad()
def getLabel(f_map, loc, device, cost_weight):
    
    batch_size, height, width = f_map.shape[0], f_map.shape[2], f_map.shape[3]
    softmax = nn.Softmax(dim=1)
    soft_map = softmax(f_map).permute(0,2,3,1).contiguous().view(batch_size, -1, 2)
    # print(soft_map.shape)
    # print(soft_map[0])
    
    a = torch.linspace(0, height-1, steps=height)
    b = torch.linspace(0, width-1, steps=width)

    x, y = torch.meshgrid(a, b)

    pd_loc = torch.stack([x.ravel(), y.ravel()], dim=1).to(device)
    
    label = torch.zeros((batch_size, height * width)).long()
    
    for i in range(batch_size):
        idx = torch.where(loc[:,2] == i)
        coordinates = loc[idx][:,:2].to(device)
        cost_dist = torch.cdist(pd_loc, coordinates, p=2)
    
        cost_class = -soft_map[i][:,1].view(-1, 1)
        
        cost_matrix = (cost_dist * cost_weight[0] + cost_class * cost_weight[1]).detach().cpu().numpy()
        
        indice = linear_sum_assignment(cost_matrix)
        # print(pd_loc[indice[0]][:20])
        # print(coordinates[indice[1]][:20])
        
        label[i][indice[0]] = 1
    
    return label
        
       

def updateStateDict(stateDict):
    stateDict.update({'layer1.0.weight':stateDict.pop("features.0.weight")})
    stateDict.update({'layer1.0.bias':stateDict.pop("features.0.bias")})
    stateDict.update({'layer2.0.weight':stateDict.pop("features.2.weight")})
    stateDict.update({'layer2.0.bias':stateDict.pop("features.2.bias")})
    stateDict.update({'layer3.0.weight':stateDict.pop("features.5.weight")})
    stateDict.update({'layer3.0.bias':stateDict.pop("features.5.bias")})
    stateDict.update({'layer4.0.weight':stateDict.pop("features.7.weight")})
    stateDict.update({'layer4.0.bias':stateDict.pop("features.7.bias")})
    stateDict.update({'layer5.0.weight':stateDict.pop("features.10.weight")})
    stateDict.update({'layer5.0.bias':stateDict.pop("features.10.bias")})
    stateDict.update({'layer6.0.weight':stateDict.pop("features.12.weight")})
    stateDict.update({'layer6.0.bias':stateDict.pop("features.12.bias")})
    stateDict.update({'layer7.0.weight':stateDict.pop("features.14.weight")})
    stateDict.update({'layer7.0.bias':stateDict.pop("features.14.bias")})
    stateDict.update({'layer8.0.weight':stateDict.pop("features.17.weight")})
    stateDict.update({'layer8.0.bias':stateDict.pop("features.17.bias")})
    stateDict.update({'layer9.0.weight':stateDict.pop("features.19.weight")})
    stateDict.update({'layer9.0.bias':stateDict.pop("features.19.bias")})
    stateDict.update({'layer10.0.weight':stateDict.pop("features.21.weight")})
    stateDict.update({'layer10.0.bias':stateDict.pop("features.21.bias")})
    stateDict.update({'layer11.0.weight':stateDict.pop("features.24.weight")})
    stateDict.update({'layer11.0.bias':stateDict.pop("features.24.bias")})
    stateDict.update({'layer12.0.weight':stateDict.pop("features.26.weight")})
    stateDict.update({'layer12.0.bias':stateDict.pop("features.26.bias")})
    stateDict.update({'layer13.0.weight':stateDict.pop("features.28.weight")})
    stateDict.update({'layer13.0.bias':stateDict.pop("features.28.bias")})
    return stateDict

def loadVGG16(myModel):
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    myModel_dict = myModel.state_dict()
    pretrained_dict = updateStateDict(pretrained_dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in myModel_dict}
    myModel_dict.update(pretrained_dict)
    myModel.load_state_dict(myModel_dict)
    
    return myModel
