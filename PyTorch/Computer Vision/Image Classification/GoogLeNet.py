'''
"Going deeper with convolutions" (Szegedy et al., 2014):
https://arxiv.org/pdf/1409.4842.pdf
'''
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Flatten, Linear, Softmax
from torch.nn.init import kaiming_normal_, normal_, constant_


class Conv_Module(Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.block = Sequential(
            Conv2d(in_channels, out_channels, **kwargs),
            BatchNorm2d(out_channels),
            ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Inception_Module(Module): 
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        self.block1 = Conv_Module(in_channels, ch1x1, kernel_size=1)
        self.block2 = Sequential(
            Conv_Module(in_channels, ch3x3red, kernel_size=1),
            Conv_Module(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.block3 = Sequential(
            Conv_Module(in_channels, ch5x5red, kernel_size=1),
            Conv_Module(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.block4 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv_Module(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        block1, block2, block3, block4 = self.block1(x), self.block2(x), self.block3(x), self.block4(x)
        return torch.cat([block1, block2, block3, block4], dim=1)


class GoogLeNet(Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Feature Extractor
        self.conv1    = Conv_Module(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2    = Conv_Module(64, 64, kernel_size=1)
        self.conv3    = Conv_Module(64, 192, kernel_size=3, padding=1),
        self.maxpool3 = MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.inception3a = Inception_Module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_Module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3    = MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception_Module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_Module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_Module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_Module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_Module(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4    = MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception_Module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_Module(832, 384, 192, 384, 48, 128, 128)
        self.avgpool     = Sequential(AdaptiveAvgPool2d((1, 1)), Flatten())
        
        # Auxiliary Classifier 1    
        self.aux1 = Sequential(AdaptiveAvgPool2d((4, 4)),
                               Conv_Module(512, 128, kernel_size=1), Flatten(),
                               Linear(2048, 1024), ReLU(), Dropout(0.7),
                               Linear(1024, num_classes), Softmax())
        
        # Auxiliary Classifier 2
        self.aux2 = Sequential(AdaptiveAvgPool2d((4, 4)),
                               Conv_Module(528, 128, kernel_size=1), Flatten(),
                               Linear(2048, 1024), ReLU(), Dropout(0.7),
                               Linear(1024, num_classes), Softmax())
        
        # Classifier
        self.fc = Sequential(Dropout(0.4), Linear(1024, num_classes), Softmax())

        # Weight Initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                constant_(module.bias, 0)
            elif isinstance(module, BatchNorm2d):
                constant_(module.weight, 1)
                constant_(module.bias, 0)
            elif isinstance(module, Linear):
                normal_(module.weight, std=0.01)
                constant_(module.bias, 0)
                    
    def forward(self, x):
        ext = self.conv1(x)
        ext = self.maxpool1(ext)
        ext = self.conv2(ext)
        ext = self.conv3(ext)
        ext = self.maxpool2(ext)
        
        ext = self.inception3a(ext)
        ext = self.inception3b(ext)
        ext = self.maxpool3(ext)
        
        ext = self.inception4a(ext)
        aux1 = self.aux1(ext)
        ext = self.inception4b(ext)
        ext = self.inception4c(ext)
        ext = self.inception4d(ext)
        aux2 = self.aux2(ext)
        ext = self.inception4e(ext)
        ext = self.maxpool4(ext)
        
        ext = self.inception5a(ext)
        ext = self.inception5b(ext)
        ext = self.avgpool(ext)
        
        cls = self.fc(ext)
        return cls, aux2, aux1 # aux losses are weighted by 0.3
        
