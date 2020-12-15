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
        self.dec3_2  = Sequential(Conv2d(128, 128, kernel_size=3), ReLU())
        
        self.upconv4 = Sequential(ConvTranspose2d(128, 64, kernel_size=2, stride=2), ReLU())
        self.dec4_1  = Sequential(Conv2d(128, 64, kernel_size=3), ReLU())
        self.dec4_2  = Sequential(Conv2d(64, 64, kernel_size=3), ReLU())
        
        self.dec_fin = Sequential(Conv2d(64, out_channels, kernel_size=1), Softmax())
        
        # Weight Initialization
        self._initialize_weights(self.enc1_1,  self.enc1_2, \
                                 self.enc2_1,  self.enc2_2, \
                                 self.enc3_1,  self.enc3_2, \
                                 self.enc4_1,  self.enc4_2, \
                                 self.b1,      self.b2, \
                                 self.upconv1, self.dec1_1, self.dec1_2, \
                                 self.upconv2, self.dec2_1, self.dec2_2, \
                                 self.upconv3, self.dec3_1, self.dec3_2, \
                                 self.upconv4, self.dec4_1, self.dec4_2, \
                                 self.dec_fin)
                                 
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                    kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        e1_conv = self.enc1_2(self.enc1_1(x))
        e1_pool = self.pool1(e1_conv)

        e2_conv = self.enc2_2(self.enc2_1(e1_pool))
        e2_pool = self.pool2(e2_conv)

        e3_conv = self.enc3_2(self.enc3_1(e2_pool))
        e3_pool = self.pool3(e3_conv)

        e4_conv = self.enc4_2(self.enc4_1(e3_pool))
        e4_pool = self.pool4(e4_conv)
        
        #Bottleneck
        b = self.b2(self.b1(e4_pool))

        # Decoder
        d1 = self.dec1_2(self.dec1_1(torch.cat([e4_conv, self.upconv1(b)],  dim=1)))
        d2 = self.dec2_2(self.dec2_1(torch.cat([e3_conv, self.upconv1(d1)], dim=1)))
        d3 = self.dec3_2(self.dec3_1(torch.cat([e2_conv, self.upconv1(d2)], dim=1)))
        d4 = self.dec4_2(self.dec4_1(torch.cat([e1_conv, self.upconv1(d3)], dim=1)))
        d_ = self.dec_fin(d4)
        return d_
