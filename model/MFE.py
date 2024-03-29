
# Multi-scale Feature Extractor
class MFE(nn.Module):
    def __init__(self, num_features_in, feature_channel=256):
        super(MFE, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_channel, feature_channel, kernel_size=3, padding=1)

        self.output = nn.Conv2d(feature_channel, 2, kernel_size=3, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.c2 = nn.Conv2d(feature_channel, 2, kernel_size=7, stride=4, padding=3)
        self.act = nn.ReLU()
   
    def forward(self, x, C5):
        out1 = self.conv1(x)
        out1 = self.act(out1)

        out1 = self.conv2(out1)
        out1 = self.act(out1)
        
        out2 = self.c2(self.maxpool(out1))
        out1 = self.output(out1)

        out1 = out1.permute(0, 2, 3, 1)
        out2 = out2.permute(0, 2, 3, 1)

        return out1.contiguous().view(out1.shape[0], -1, 2), out2.contiguous().view(out2.shape[0], -1, 2)