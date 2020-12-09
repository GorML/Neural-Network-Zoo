"""
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation (Badrinarayanan et al., 2016): https://arxiv.org/pdf/1511.00561.pdf
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, MaxPool2d, MaxUnpool2d, Softmax
from torch.nn.init import kaiming_normal_, zeros_, ones_


class SegNet(Module):
    def __init__(self, in_channels=3, num_classes=11):
        super().__init__()

        # Encoder
        self.enc_conv11 = Sequential(Conv2d(in_channels, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        self.enc_conv12 = Sequential(Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        self.pool1      = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_conv21 = Sequential(Conv2d(64, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        self.enc_conv22 = Sequential(Conv2d(128, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        self.pool2      = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_conv31 = Sequential(Conv2d(128, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.enc_conv32 = Sequential(Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.enc_conv33 = Sequential(Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.pool3      = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_conv41 = Sequential(Conv2d(256, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.enc_conv42 = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.enc_conv43 = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.pool4      = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_conv51 = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.enc_conv52 = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.enc_conv53 = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.pool5      = MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder
        self.unpool1    = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv11 = Sequential(ConvTranspose2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.dec_conv12 = Sequential(ConvTranspose2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.dec_conv13 = Sequential(ConvTranspose2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        
        self.unpool2    = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv21 = Sequential(ConvTranspose2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.dec_conv22 = Sequential(ConvTranspose2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.dec_conv23 = Sequential(ConvTranspose2d(512, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        
        self.unpool3    = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv31 = Sequential(ConvTranspose2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.dec_conv32 = Sequential(ConvTranspose2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.dec_conv33 = Sequential(ConvTranspose2d(256, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        
        self.unpool4    = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv41 = Sequential(ConvTranspose2d(128, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        self.dec_conv42 = Sequential(ConvTranspose2d(128, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        
        self.unpool5    = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv51 = Sequential(ConvTranspose2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())   
        self.dec_conv52 = Sequential(ConvTranspose2d(64, num_classes, kernel_size=3, padding=1), Softmax())
        
        # Weight Initialization
        self._initialize_weights(self.enc_conv0, self.enc_conv1, self.enc_conv2, self.enc_conv3,
                                 self.dec_conv0, self.dec_conv1, self.dec_conv2, self.dec_conv3)
        
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                    kaiming_normal_(module.weight)
                    zeros_(module.bias)
                elif isinstance(module, BatchNorm2d):
                    ones_(module.weight)
                    zeros_(module.bias)

    def forward(self, x):
        # Encoder
        e1, e1_idx = self.pool1(self.enc_conv1(x))
        e2, e2_idx = self.pool2(self.enc_conv2(e1))
        e3, e3_idx = self.pool3(self.enc_conv3(e2))
        e4, e4_idx = self.pool4(self.enc_conv4(e3))
        e5, e5_idx = self.pool4(self.enc_conv4(e4))

        # Decoder
        d1 = self.dec_conv1(self.unpool1(e5, e5_idx))
        d2 = self.dec_conv2(self.unpool2(d1, e4_idx))
        d3 = self.dec_conv3(self.unpool3(d2, e3_idx))
        d4 = self.dec_conv4(self.unpool4(d3, e2_idx))
        d5 = self.dec_conv4(self.unpool4(d4, e1_idx))
        return d5
