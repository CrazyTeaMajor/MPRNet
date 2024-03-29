
# Regional Maximum Replacement and 1v1 Points Match 
class Matcher(nn.Module):
    def __init__(self, coef_class=1, coef_location=0.05):
        super(Matcher, self).__init__()
        self.coef_class = coef_class
        self.coef_loc = coef_location
        
    def forward(self, feature_map, c2, pre_point, label, height, width):
        bs = len(feature_map)
        indices = []
        gap = 8
        for i in range(bs):
            f_map = feature_map[i].softmax(-1)
            f_map2 = f_map[:,1].reshape(height, width)
            b = c2[i].softmax(-1)
            b = b[:,1].reshape(height // gap, width // gap)
            
            for u in range(b.shape[0]):
                for v in range(b.shape[1]):
                    idx = f_map2[u*gap:u*gap+gap, v*gap:v*gap+gap].argmax()
                    x = idx // gap
                    y = idx - gap * x
                    if f_map2[u*gap+x,v*gap+y] < 0.5 and b[u,v] > 0.5:
                        f_map2[u * gap + x, v * gap + y] = b[u,v]
 			# f_map2[u*gap+x,v*gap+y] = max(f_map2[u*gap+x,v*gap+y], b[u,v])		
                        # feature_map[i][(u*gap+x)*width + v*gap+y, :] = c2[i][u*width//gap + v, :]
            
            f_map2 = f_map2.reshape(height * width, 1)
            f_map[:,1] = f_map2.view(-1)
            label_point = label[i]['point']
            cost_class = -f_map[:, 1].view(-1, 1) 
            
            if len(label_point) > 0:
                cost_point = torch.cdist(pre_point[i], label_point, p=2) 
            else:
                cost_point = 0.
        
            C = self.coef_loc * cost_point + self.coef_class * cost_class
            indice = linear_sum_assignment(C.detach().cpu().numpy())
            indices.append(indice)
           
        return indices, feature_map


