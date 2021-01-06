'''
"Deep Residual Learning for Image Recognition" (He et al., 2015):
https://arxiv.org/pdf/1512.03385.pdf
"Identity Mappings in Deep Residual Networks" (He et al., 2016):
https://arxiv.org/pdf/1603.05027v3.pdf
'''
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, GroupNorm, ReLU, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear
from torch.nn.init import kaiming_normal_, constant_


class ResidualBlock(Module):
    
    expansion = 4
    
    def __init__(self, in_channels, conv_channels, stride=1, downsample=None, base_width=64):
        super().__init__()
        
        width = int(in_channels * (base_width / 64))
        # NVIDIA's ResNet V1.5: stride for downsampling is at the first 1x1 convolution instead of the 3x3 convolution.
        self.conv = Sequential(BatchNorm2d(width), ReLU(), Conv2d(in_channels, width, kernel_size=1),
                               BatchNorm2d(width), ReLU(), Conv2d(width, width, kernel_size=3, stride, padding=1),
                               BatchNorm2d(conv_channels * self.expansion), Conv2d(width, conv_channels * self.expansion, kernel_size=1))
        self.last_relu = ReLU()

    def forward(self, x):
        # Identity Shortcut
        identity = x
        # Residual Block with a BN & ReLU pre-activation instead of post-activation.
        conv = self.conv(x)
        # Identity Mapping
        conv += identity
        conv = self.last_relu(conv)
        return conv


class ResNet(Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        """
        Layers for:
        - Resnet-50:  [3, 4, 6, 3]
        - Resnet-101: [3, 4, 23, 3]
        - Resnet-152: [3, 8, 36, 3]
        """
        self.in_channels = 64
        
        self.convblock = Sequential(Conv2d(3, self.conv_channels, kernel_size=7, stride=2, padding=3), BatchNorm2d(self.conv_channels), ReLU())
        self.maxpool   = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, layers[0])
        self.layer2 = self._make_layer(ResidualBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d(1)
        
        self.fc = Sequential(Flatten(), Linear(512 * ResidualBlock.expansion, num_classes))
        
        # Weight Initialization
        for module in self.modules(): 
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                constant_(module.bias, 0)
            elif isinstance(module, (BatchNorm2d, GroupNorm)):
                constant_(module.weight, 1)
                constant_(module.bias, 0)

    def _make_layer(self, conv_channels, blocks, stride=1):
        if stride != 1 or self.in_channels != conv_channels * ResidualBlock.expansion:
            downsample = Sequential(
                Conv2d(self.in_channels, conv_channels * ResidualBlock.expansion, kernel_size=1, stride),
                BatchNorm2d(conv_channels * ResidualBlock.expansion)
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, conv_channels, stride, downsample))
        self.in_channels = conv_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, conv_channels))
        return Sequential(*layers)

    def forward(self, x):
        # Feature Space Transformer
        conv = self.maxpool(self.convblock(x))
        l1 = self.layer1(conv)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        ext = self.avgpool(l4)
        
        # Classifier
        cls = self.fc(ext)
        return cls
