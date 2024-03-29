# loading images from specific datasets
class Dataset(Dataset):
    def __init__(self, img_root, train = True):
        self.img_root = img_root
        self.img_path_list = []
        self.train = train
        
        name_list = os.listdir(img_root)
        
        for i in range(len(name_list)):
            if name_list[i].endswith(".jpg"):
                img_path = os.path.join(img_root, name_list[i])
                self.img_path_list.append(img_path)
        
        self.nSamples = len(self.img_path_list)
        
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        
        img = Image.open(self.img_path_list[index]).convert('RGB')
        points = []
        
        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                       ])
        
        img_path = self.img_path_list[index]
        
        # load differrnt dataset
        if self.img_root.find('SHHA') != -1:
            gt_path = img_path.replace("images", "ground_truth").replace("IMG","GT_IMG").replace(".jpg",".mat")
            gt = scio.loadmat(gt_path)
            gt_map = gt["image_info"][0][0][0][0][0]

            for i in range(len(gt_map)):
                points.append([gt_map[i][0], gt_map[i][1]])
        else:
            gt_path = img_path.replace('images', 'gt').replace('.jpg','.txt')

            with open(gt_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.split()
                    x = float(data[0])
                    y = float(data[1])
                    points.append([x, y])
            
        img = transform(img)
        points = np.array(points)
        
        if self.train:
            size = 192
            num_patch = 4
            # random resize
            scale_range = [0.8, 1.2]
            scale = random.uniform(*scale_range)
            if min(img.size(1), img.size(2)) * scale < size:
                scale = 2
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            points *= scale
            
            # random crop
            img, points = crop(img, points, size, num_patch)
            
            # random flip
            if random.random() > 0.5:
                img = np.array(img)
                img = torch.Tensor(img[:, :, :, ::-1].copy())
                for i in range(len(points)):
                    if len(points[i]) > 0:
                        points[i][:, 0] = size - points[i][:, 0]
                        points[i] = torch.FloatTensor(points[i])
        else:
            scale = 2000 * 2000 / (img.shape[1] * img.shape[2])
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=min(1,scale)).squeeze(0)
          
            # new_size = [int(img.shape[1] / scale) // 32 * 32, int(img.shape[2] / scale) // 32 * 32]
            new_h = img.shape[1]
            new_w = img.shape[2]
            if img.shape[1] % 32 != 0:
                new_h = (img.shape[1] // 32 + 1) * 32
                
            if img.shape[2] % 32 != 0:
                new_w = (img.shape[2] // 32 + 1) * 32
                
            new_img = torch.zeros((3, new_h, new_w))
            new_img[:, :img.shape[1], :img.shape[2]] = img
            img = new_img
            
        return img, points
    
    
def crop(img, point, size, num_patch=4):
    
    height = size
    width = size
    
    imgs = np.zeros([num_patch, img.shape[0], height, width])
    points = []
    
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - height)
        start_w = random.randint(0, img.size(2) - width)

        end_h = start_h + height
        end_w = start_w + width

        imgs[i] = img[:, start_h:end_h, start_w:end_w]
        
        if len(point) == 0:
            new_point = []
        else:
            idx = (point[:, 0] >= start_w) & (point[:, 0] <= end_w) & (point[:, 1] >= start_h) & (point[:, 1] <= end_h)
            new_point = point[idx]

            new_point[:, 0] -= start_w
            new_point[:, 1] -= start_h
        
        points.append(new_point)
        
    return torch.Tensor(imgs), points