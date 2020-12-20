'''
"ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012):
http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
'''
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Softmax
from torch.nn.init import normal_, constant_


class AlexNet(Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Feature Extractor
        self.extractor = Sequential(
            Conv2d(in_channels, 96, kernel_size=11, stride=4), ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            
            Conv2d(96, 256, kernel_size=5, padding=2), ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            
            Conv2d(256, 384, kernel_size=3, padding=1), ReLU(),
            Conv2d(384, 384, kernel_size=3, padding=1), ReLU(),
            Conv2d(384, 256, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=3, stride=2), Flatten()
        )
        
        # Classifier
        self.classifier = Sequential(
            Linear(256 * 6 * 6, 4096), ReLU(),
            Dropout(),
            Linear(4096, 4096), ReLU(),
            Dropout(),
            Linear(4096, num_classes), Softmax()
        )
        
        # Weight Initialization
        for module in self.modules():
            if isinstance(module, (Conv2d, Linear)):
                normal_(module.weight, std=0.01)
                constant_(module.bias, 1)
        
    def forward(self, x):
        ext = self.extractor(x)
        cls = self.classifier(ext)
        return cls
