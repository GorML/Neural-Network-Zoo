"""
"SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation" (Badrinarayanan et al., 2016):
https://arxiv.org/pdf/1511.00561.pdf
"""
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, MaxUnpool2d, Softmax
from torch.nn.init import kaiming_normal_, constant_


class SegNet(Module):
    def __init__(self, in_channels=3, out_channels=11):
        super().__init__()

        # Encoder
        self.enc1  = Sequential(Conv2d(in_channels, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(),
                                Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc2  = Sequential(Conv2d(64, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU(),
                                Conv2d(128, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc3  = Sequential(Conv2d(128, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(),
                                Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(),
                                Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.pool3 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc4  = Sequential(Conv2d(256, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.pool4 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc5  = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.pool5 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder
        self.unpool1 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1    = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                  Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                  Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        
        self.unpool2 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2    = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                  Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(),
                                  Conv2d(512, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        
        self.unpool3 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3    = Sequential(Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(),
                                  Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(),
                                  Conv2d(256, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        
        self.unpool4 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4    = Sequential(Conv2d(128, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU(),
                                  Conv2d(128, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        
        self.unpool5 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec5    = Sequential(Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(),
                                  Conv2d(64, out_channels, kernel_size=3, padding=1), Softmax())
        
        # Weight Initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                constant_(module.bias, 0)
            elif isinstance(module, BatchNorm2d):
                constant_(module.weight, 1)
                constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        e1, e1_idx = self.pool1(self.enc1(x))
        e2, e2_idx = self.pool2(self.enc2(e1))
        e3, e3_idx = self.pool3(self.enc3(e2))
        e4, e4_idx = self.pool4(self.enc4(e3))
        e5, e5_idx = self.pool5(self.enc5(e4))

        # Decoder
        d1 = self.dec1(self.unpool1(e5, e5_idx))
        d2 = self.dec2(self.unpool2(d1, e4_idx))
        d3 = self.dec3(self.unpool3(d2, e3_idx))
        d4 = self.dec4(self.unpool4(d3, e2_idx))
        d5 = self.dec5(self.unpool5(d4, e1_idx))
        return d5
