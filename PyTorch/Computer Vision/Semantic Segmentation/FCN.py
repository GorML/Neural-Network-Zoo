"""
"Fully Convolutional Networks for Semantic Segmentation" (Long et al., 2015):
https://arxiv.org/pdf/1411.4038.pdf
"""
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d
from torch.init import zero_


class FCN8s(Module):
    def __init__(self, in_channels=3, out_channels=33):
        super().__init__()

        # Feature Extractor
        self.conv1 = Sequential(Conv2d(in_channels, 64, kernel_size=3, padding=100), ReLU(),
                                Conv2d(64, 64, kernel_size=3, padding=1), ReLU())
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Sequential(Conv2d(64, 128, kernel_size=3, padding=1), ReLU(),
                                Conv2d(128, 128, kernel_size=3, padding=1), ReLU())
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Sequential(Conv2d(128, 256, kernel_size=3, padding=1), ReLU(),
                                Conv2d(256, 256, kernel_size=3, padding=1), ReLU(),
                                Conv2d(256, 256, kernel_size=3, padding=1), ReLU())
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = Sequential(Conv2d(256, 512, kernel_size=3, padding=1), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), ReLU())
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = Sequential(Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
                                Conv2d(512, 512, kernel_size=3, padding=1), ReLU())
        self.pool5 = MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.fc = Sequential(Conv2d(512, 4096, kernel_size=7), ReLU(), Dropout2d(),
                             Conv2d(4096, 4096, kernel_size=1), ReLU(), Dropout2d(),
                             Conv2d(4096, out_channels, kernel_size=1))

        # Upscaler with Deconvolutions
        self.upscore2 = ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False)
        self.score_pool4 = Conv2d(512, out_channels, kernel_size=1)
        self.upscore_pool4 = ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False)
        self.score_pool3 = Conv2d(256, out_channels, kernel_size=1)
        self.upscore8 = ConvTranspose2d(out_channels, out_channels, kernel_size=16, stride=8, bias=False)

        # Weight Initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                module.weight.zero_()
                module.bias.zero_()
        # ConvTranspose2d initialized with
        # pretrained weights from FCN32s

    def forward(self, x):
        ext1 = self.pool1(self.conv1(x))
        ext2 = self.pool2(self.conv2(ext1))
        ext3 = self.pool3(self.conv3(ext2))
        ext4 = self.pool4(self.conv4(ext3))
        ext5 = self.pool5(self.conv5(ext4))

        cls = self.fc(ext5)

        upscore2 = self.upscore2(cls)
        score_pool4 = self.score_pool4(ext4)
        score_pool4c = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4c)

        score_pool3 = self.score_pool3(ext3)
        score_pool3c = score_pool3[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        upscore8 = self.upscore8(upscore_pool4 + score_pool3c)
        upscore8c = upscore8[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]]
        return upscore8c
