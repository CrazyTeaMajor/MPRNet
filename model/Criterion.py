
# Calculate loss 
class set_criterion(nn.Module):
    def __init__(self, num_classes=1, empty_weight=0.5, batch_size=8, device=None):
        super(set_criterion, self).__init__()
        self.criterion_mse = nn.MSELoss(reduction = 'mean').to(device)
        self.weight_dict = torch.ones(num_classes + 1).to(device)
        self.weight_dict[0] = empty_weight
        self.device = device
        
    def get_label(self, bs, num_queries, indices):
        label = []
        for i in range(bs):
            indice = indices[i][0]               
            c = torch.zeros(num_queries, dtype=torch.int64, device=self.device)
            c[indice] = 1
            label.append(c)
        return torch.stack(label, dim=-1).to(self.device)

    def get_points(self, bs, pre_points, indices, label):
        points = []
        label_points = []
        num = 0
        for i in range(bs):
            indice = indices[i]
            if len(label[i]['point']) > 0:
                points.append(pre_points[i][indice[0]])
                label_points.append(label[i]['point'][indice[1]])
                num += len(indice[1])
            else:
                points.append(torch.zeros(1,2).to(self.device))
                label_points.append(torch.zeros(1,2).to(self.device))
        return torch.cat(points), torch.cat(label_points), num
        
    def forward(self, feature_map, pre_points, indices, label):
        bs, num_queries, _ = feature_map.shape
        label_class = self.get_label(bs, num_queries, indices)
        
        feature_map = feature_map.transpose(1,2)
        label_class = label_class.transpose(0,1)
        
        pred_points, label_points, num = self.get_points(bs, pre_points, indices, label)
        loss_mse = F.mse_loss(pred_points, label_points, reduction='none')
        
        if num > 0:
            loss_mse = loss_mse.sum() / num
        else:
            loss_mse = loss_mse.sum()
        
        loss_ce = F.cross_entropy(feature_map, label_class, self.weight_dict)
        
        return loss_ce, loss_mse

