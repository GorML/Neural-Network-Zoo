"""
"U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015):
https://arxiv.org/pdf/1505.04597.pdf
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid
from torch.nn.init import kaiming_normal_, constant_


class UNet(Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        # Encoder
        self.enc1_1 = Sequential(Conv2d(in_channels, 64, kernel_size=3), ReLU())
        self.enc1_2 = Sequential(Conv2d(64, 64, kernel_size=3), ReLU())
        self.pool1  = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2_1 = Sequential(Conv2d(64, 128, kernel_size=3), ReLU())
        self.enc2_2 = Sequential(Conv2d(128, 128, kernel_size=3), ReLU())
        self.pool2  = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3_1 = Sequential(Conv2d(128, 256, kernel_size=3), ReLU())
        self.enc3_2 = Sequential(Conv2d(256, 256, kernel_size=3), ReLU())
        self.pool3  = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4_1 = Sequential(Conv2d(256, 512, kernel_size=3), ReLU())
        self.enc4_2 = Sequential(Conv2d(512, 512, kernel_size=3), ReLU())
        self.pool4  = MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.b1 = Sequential(Conv2d(512, 1024, kernel_size=3), ReLU())
        self.b2 = Sequential(Conv2d(1024, 1024, kernel_size=3), ReLU())

        # Decoder
        self.upconv1 = Sequential(ConvTranspose2d(1024, 512, kernel_size=2, stride=2), ReLU())
        self.dec1_1  = Sequential(Conv2d(1024, 512, kernel_size=3), ReLU())
        self.dec1_2  = Sequential(Conv2d(512, 512, kernel_size=3), ReLU())
        
        self.upconv2 = Sequential(ConvTranspose2d(512, 256, kernel_size=2, stride=2), ReLU())
        self.dec2_1  = Sequential(Conv2d(512, 256, kernel_size=3), ReLU())
        self.dec2_2  = Sequential(Conv2d(256, 256, kernel_size=3), ReLU())
        
        self.upconv3 = Sequential(ConvTranspose2d(256, 128, kernel_size=2, stride=2), ReLU())
        self.dec3_1  = Sequential(Conv2d(256, 128, kernel_size=3), ReLU())
        self.dec3_1  = Sequential(Conv2d(128, 128, kernel_size=3), ReLU())
        
        self.upconv4 = Sequential(ConvTranspose2d(128, 64, kernel_size=2, stride=2), ReLU())
        self.dec4_1  = Sequential(Conv2d(128, 64, kernel_size=3), ReLU())
        self.dec4_2  = Sequential(Conv2d(64, 64, kernel_size=3), ReLU())
        
        self.dec_fin = Sequential(Conv2d(64, out_channels, kernel_size=1), Sigmoid())
        
        # Weight Initialization
        self._initialize_weights(self.enc1_1,  self.enc1_2, \
                                 self.enc2_1,  self.enc2_2, \
                                 self.enc3_1,  self.enc3_2, \
                                 self.enc4_1,  self.enc4_2, \
                                 self.b1,          self.b2, \
                                 self.dec_upconv1, self.dec1_1, self.dec1_2, \
                                 self.dec_upconv2, self.dec2_1, self.dec2_2, \
                                 self.dec_upconv3, self.dec3_1, self.dec3_2, \
                                 self.dec_upconv4, self.dec4_1, self.dec4_2, \
                                 self.dec_fin)
                                 
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                    kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.pool1(self.enc1_2(self.enc1_1(x)))
        e2 = self.pool2(self.enc2_2(self.enc2_1(e1)))
        e3 = self.pool3(self.enc3_2(self.enc3_1(e2)))
        e4 = self.pool4(self.enc4_2(self.enc4_1(e3)))
        
        #Bottleneck
        b = self.b2(self.b1(e4))

        # Decoder
        d1 = self.dec1_2(self.dec1_1(self.upconv1(b)))
        d2 = self.dec2_2(self.dec2_1(self.upconv2(d1)))
        d3 = self.dec3_2(self.dec3_1(self.upconv3(d2)))
        d4 = self.dec4_2(self.dec4_1(self.upconv4(d3)))
        d_ = self.dec_fin(d4)
        return d_
