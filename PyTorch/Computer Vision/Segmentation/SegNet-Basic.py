"""
SegNet Architecture (2016): https://arxiv.org/pdf/1511.00561.pdf
Kaiming Weight Initialization (2015): https://arxiv.org/pdf/1502.01852.pdf
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, MaxPool2d, MaxUnpool2d, Softmax
from torch.nn.init import kaiming_normal_, zeros_, ones_


class SegNet_Basic(Module):
    def __init__(self, in_channels=3, num_classes=11):
        super().__init__()

        # Encoder
        self.enc_conv1 = Sequential(Conv2d(in_channels, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv2 = Sequential(Conv2d(64, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv3 = Sequential(Conv2d(128, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.pool3 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv4 = Sequential(Conv2d(256, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU())
        self.pool4 = MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        self.bottleneck = Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder
        self.unpool1 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = Sequential(ConvTranspose2d(512, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU())
        self.unpool2 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = Sequential(ConvTranspose2d(256, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU())
        self.unpool3 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = Sequential(ConvTranspose2d(128, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU())
        self.unpool4 = MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv4 = Sequential(ConvTranspose2d(64, num_classes, kernel_size=3, padding=1), Softmax())
        
        # Weight Initialization
        self._initialize_weights(self.enc_conv0, self.enc_conv1, self.enc_conv2, self.enc_conv3,
                                 self.dec_conv0, self.dec_conv1, self.dec_conv2, self.dec_conv3)
        
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, Conv2d):
                    kaiming_normal_(module.weight)
                    if module.bias is not None:
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

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d1 = self.dec_conv1(self.unpool1(b,  e4_idx))
        d2 = self.dec_conv2(self.unpool2(d1, e3_idx))
        d3 = self.dec_conv3(self.unpool3(d2, e2_idx))
        d4 = self.dec_conv4(self.unpool4(d3, e1_idx))
        return d4
