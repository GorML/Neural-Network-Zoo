"""
U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015): https://arxiv.org/pdf/1505.04597.pdf
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid
from torch.nn.init import kaiming_normal_, zeros_

class UNet(Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # Encoder
        self.enc_conv11 = Sequential(Conv2d(in_channels, 64, kernel_size=3), ReLU())
        self.enc_conv12 = Sequential(Conv2d(64, 64, kernel_size=3), ReLU())
        self.pool1      = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv21 = Sequential(Conv2d(64, 128, kernel_size=3), ReLU())
        self.enc_conv22 = Sequential(Conv2d(128, 128, kernel_size=3), ReLU())
        self.pool2      = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv31 = Sequential(Conv2d(128, 256, kernel_size=3), ReLU())
        self.enc_conv32 = Sequential(Conv2d(256, 256, kernel_size=3), ReLU())
        self.pool3      = MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_conv41 = Sequential(Conv2d(256, 512, kernel_size=3), ReLU())
        self.enc_conv42 = Sequential(Conv2d(512, 512, kernel_size=3), ReLU())
        self.pool4      = MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.b1         = Sequential(Conv2d(512, 1024, kernel_size=3), ReLU())
        self.b2         = Sequential(Conv2d(1024, 1024, kernel_size=3), ReLU())

        # Decoder
        self.upconv1    = Sequential(ConvTranspose2d(1024, 512, kernel_size=2, stride=2), ReLU())
        self.dec_conv11 = Sequential(Conv2d(1024, 512, kernel_size=3), ReLU())
        self.dec_conv12 = Sequential(Conv2d(512, 512, kernel_size=3), ReLU())
        
        self.upconv2    = Sequential(ConvTranspose2d(512, 256, kernel_size=2, stride=2), ReLU())
        self.dec_conv21 = Sequential(Conv2d(512, 256, kernel_size=3), ReLU())
        self.dec_conv22 = Sequential(Conv2d(256, 256, kernel_size=3), ReLU())
        
        self.upconv3    = Sequential(ConvTranspose2d(256, 128, kernel_size=2, stride=2), ReLU())
        self.dec_conv31 = Sequential(Conv2d(256, 128, kernel_size=3), ReLU())
        self.dec_conv31 = Sequential(Conv2d(128, 128, kernel_size=3), ReLU())
        
        self.upconv4    = Sequential(ConvTranspose2d(128, 64, kernel_size=2, stride=2), ReLU())
        self.dec_conv41 = Sequential(Conv2d(128, 64, kernel_size=3), ReLU())
        self.dec_conv42 = Sequential(Conv2d(64, 64, kernel_size=3), ReLU())
        
        self.dec_last   = Sequential(ConvTranspose2d(64, num_classes, kernel_size=1), Sigmoid())
        
        # Weight Initialization
        self._initialize_weights(self.enc_conv11,  self.enc_conv12, \
                                 self.enc_conv21,  self.enc_conv22, \
                                 self.enc_conv31,  self.enc_conv32, \
                                 self.enc_conv41,  self.enc_conv42, \
                                 self.b1,          selfb2, \
                                 self.dec_upconv1, self.dec_conv11, self.dec_conv12, \
                                 self.dec_upconv2, self.dec_conv21, self.dec_conv22, \
                                 self.dec_upconv3, self.dec_conv31, self.dec_conv32, \
                                 self.dec_upconv4, self.dec_conv41, self.dec_conv42, \
                                 self.dec_last)
                                 
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                    kaiming_normal_(module.weight)
                    zeros_(module.bias)

    def forward(self, x):
        # Encoder
        e1 = self.pool1(self.enc_conv12(self.enc_conv11(x)))
        e2 = self.pool2(self.enc_conv22(self.enc_conv21(e1)))
        e3 = self.pool3(self.enc_conv32(self.enc_conv31(e2)))
        e4 = self.pool4(self.enc_conv42(self.enc_conv41(e3)))
        
        #Bottleneck
        b = self.b2(self.b1(e4))

        # Decoder
        d1 = self.dec_conv12(self.dec_conv11(self.upconv1(b)))
        d2 = self.dec_conv22(self.dec_conv21(self.upconv2(d1)))
        d3 = self.dec_conv32(self.dec_conv31(self.upconv3(d2)))
        d4 = self.dec_conv42(self.dec_conv41(self.upconv4(d3)))
        d_ = self.dec_last(d4)
        return d_
