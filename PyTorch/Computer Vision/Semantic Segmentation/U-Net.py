"""
"U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015):
https://arxiv.org/pdf/1505.04597.pdf
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d, Softmax
from torch.nn.init import kaiming_normal_, constant_


class UNet(Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        # Encoder
        self.enc1  = Sequential(Conv2d(in_channels, 64, kernel_size=3), ReLU(),
                                Conv2d(64, 64, kernel_size=3), ReLU())
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2  = Sequential(Conv2d(64, 128, kernel_size=3), ReLU(),
                               Conv2d(128, 128, kernel_size=3), ReLU())
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3  = Sequential(Conv2d(128, 256, kernel_size=3), ReLU(),
                                Conv2d(256, 256, kernel_size=3), ReLU())
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4  = Sequential(Conv2d(256, 512, kernel_size=3), ReLU(),
                                Conv2d(512, 512, kernel_size=3), ReLU())
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.b = Sequential(Conv2d(512, 1024, kernel_size=3), ReLU(),
                            Conv2d(1024, 1024, kernel_size=3), ReLU())

        # Decoder
        self.upconv1 = ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1    = Sequential(Conv2d(1024, 512, kernel_size=3), ReLU(),
                                  Conv2d(512, 512, kernel_size=3), ReLU())
        
        self.upconv2 = ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2    = Sequential(Conv2d(512, 256, kernel_size=3), ReLU(),
                                  Conv2d(256, 256, kernel_size=3), ReLU())
        
        self.upconv3 = ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3    = Sequential(Conv2d(256, 128, kernel_size=3), ReLU(),
                                  Conv2d(128, 128, kernel_size=3), ReLU())
        
        self.upconv4 = ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4    = Sequential(Conv2d(128, 64, kernel_size=3), ReLU(),
                                  Conv2d(64, 64, kernel_size=3), ReLU(),
                                  Conv2d(64, out_channels, kernel_size=1), Softmax())
        
        # Weight Initialization
        for module in self.modules():
            if isinstance(module, (Conv2d, ConvTranspose2d)):
                kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        e1_conv = self.enc1(x)
        e1_pool = self.pool1(e1_conv)

        e2_conv = self.enc2(e1_pool)
        e2_pool = self.pool2(e2_conv)

        e3_conv = self.enc3(e2_pool)
        e3_pool = self.pool3(e3_conv)

        e4_conv = self.enc4(e3_pool)
        e4_pool = self.pool4(e4_conv)
        
        #Bottleneck
        b = self.b(e4_pool)

        # Decoder
        d1 = self.dec1(torch.cat([e4_conv, self.upconv1(b)],  dim=1))
        d2 = self.dec2(torch.cat([e3_conv, self.upconv2(d1)], dim=1))
        d3 = self.dec3(torch.cat([e2_conv, self.upconv3(d2)], dim=1))
        d4 = self.dec4(torch.cat([e1_conv, self.upconv4(d3)], dim=1))
        return d4
