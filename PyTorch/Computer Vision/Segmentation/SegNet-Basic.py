from torch.nn import Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, MaxPool2d, MaxUnpool2d, Softmax
from torch.nn.init import kaiming_normal_, zeros_, ones_


class SegNet_Basic(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # Encoder
        self.enc_conv0 = Sequential(Conv2d(in_channels, 64, kernel_size=3, padding=1),
                                       BatchNorm2d(64),
                                       ReLU())
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder
        self.unpool0 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv0 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU())
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(nn.ConvTranspose2d(64, num_classes, kernel_size=3, padding=1),
                                       nn.Softmax())
        # Weight Initialization
        self._initialize_weights(self.enc_conv0, self.enc_conv1, self.enc_conv2, self.enc_conv3,
                                 self.dec_conv0, self.dec_conv1, self.dec_conv2, self.dec_conv3)
        
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    kaiming_normal_(module.weight)
                    if module.bias is not None:
                        zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    ones_(module.weight)
                    zeros_(module.bias)

    def forward(self, x):
        # Encoder
        e0, e0_idx = self.pool0(self.enc_conv0(x))
        e1, e1_idx = self.pool1(self.enc_conv1(e0))
        e2, e2_idx = self.pool2(self.enc_conv2(e1))
        e3, e3_idx = self.pool3(self.enc_conv3(e2))

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d0 = self.dec_conv0(self.unpool0(b,  e3_idx))
        d1 = self.dec_conv1(self.unpool1(d0, e2_idx))
        d2 = self.dec_conv2(self.unpool2(d1, e1_idx))
        d3 = self.dec_conv3(self.unpool3(d2, e0_idx))
        return d3
