'''
"Deep Residual Learning for Image Recognition" (He et al., 2015):
https://arxiv.org/pdf/1512.03385.pdf
'''
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, GroupNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Flatten, Linear, Softmax
from torch.nn.init import kaiming_normal_, constant_


def Conv_Block(in_channels, out_channels, **kwargs):
    return Sequential(
        Conv2d(in_channels, out_channels, **kwargs),
        BatchNorm2d(out_channels),
        ReLU()
    )


class Bottleneck(Module):
    def __init__(self, in_channels, conv_channels, stride, groups, base_width=64, expansion=4):
        super().__init__()
        
        width = int(in_channels * (base_width / 64)) * groups
        # NVIDIA's ResNet V1.5: stride for downsampling is at the first 1x1 convolution instead of the 3x3 convolution.
        self.conv1   = Conv_Block(in_channels, width, kernel_size=1)
        self.conv2   = Conv_Block(width, width, kernel_size=3, stride, padding=1, dilation=1, groups=groups)
        self.conv3   = Conv2d(width, conv_channels * expansion, kernel_size=1)
        self.last_bn = BatchNorm2d(conv_channels * expansion)

    def forward(self, x):
        # Identity Shortcut
        identity = x
        # Feature Extractor
        conv = self.last_bn(self.conv3(self.conv2(self.conv1(x))))
        # Identity Mapping
        conv += identity
        conv = self.relu(conv)
        return conv


class ResNet(Module):
    def __init__(self, block=Bottleneck, layers, num_classes=1000, groups=1, width_per_group=64):
        super().__init__()
        
        self.in_channels = 64
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv    = Conv_Block(3, self.conv_channels, kernel_size=7, stride=2, padding=3)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = Sequential(AdaptiveAvgPool2d((1, 1)), Flatten())
        
        self.fc = Linear(512 * block.expansion, num_classes)

        """
        Zero-initialize the last BatchNorm2d in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Goyal et al., 2018):
        https://arxiv.org/pdf/1706.02677.pdf
        """
        for module in self.modules(): 
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                constant_(module.bias, 0)
            elif isinstance(module, (BatchNorm2d, GroupNorm)):
                constant_(module.weight, 1)
                constant_(module.bias, 0)
            elif isinstance(module, Bottleneck):
                constant_(module.last_bn.weight, 0)

    def _make_layer(self, block=Bottleneck, conv_channels, blocks, stride=1):
        if stride != 1 or self.in_channels != conv_channels * block.expansion:
            downsample = Sequential(
                Conv2d(self.in_channels, conv_channels * block.expansion, kernel_size=1, stride),
                BatchNorm2d(channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, conv_channels, stride, downsample, self.groups, self.base_width))
        self.in_channels = conv_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, conv_channels, groups=self.groups, base_width=self.base_width))
        return Sequential(*layers)

    def forward(self, x):
        # Feature Extractor
        conv = self.maxpool(self.conv(x))
        l1 = self.layer1(conv)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        ext = self.avgpool(l4)
        # Classifier
        cls = self.fc(ext)
        return cls
