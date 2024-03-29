
# Multi-scale Pointwise Regression Network
class MPRNet(nn.Module):
    def __init__(self, Matcher=None):
        super(MPRNet, self).__init__()
        
        self.Matcher = Matcher
        
        vgg16_bn = models.vgg16_bn(pretrained=True)
        features = list(vgg16_bn.features.children())
        
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])
        
        feature_size = 256
        
        self.P5_1 = nn.Conv2d(512, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(512, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(256, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.Multi_scale_feature_extractor = MFE(num_features_in=feature_size, feature_channel=256)
        
        
    def forward(self, x, label=None):
        bs = len(x)

        body1 = self.body1(x)
        body2 = self.body2(body1)
        body3 = self.body3(body2)
        body4 = self.body4(body3)
        
        C3, C4, C5 = [body2, body3, body4]
        
        P5_x = self.P5_1(C5)         # shape = [bs, 256, /16, /16]
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)       # shape = [bs, 256, /8, /8]

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)       # shape = [bs, 256, /4, /4]
        
        P4_x = self.P4_upsampled(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)       # shape = [bs, 256, /4, /4]
        
        feature_map_level_4 = P4_x
        feature_map_level_5 = C5
        
        bs, _, height, width = feature_map_level_4.shape
        pointwise_map_1, pointwise_map_2 = self.Multi_scale_feature_extractor(feature_map_level_4,  feature_map_level_5)
        
        if label is not None:
            pre_points = get_pre_points(height, width).cuda().repeat(bs, 1, 1)
            indices, pointwise_map_3 = self.matcher(pointwise_map_1, pointwise_map_2, pre_points, label, height, width)
            return pointwise_map_3, pre_points, indices
        else:
            return pointwise_map_3, c2





            