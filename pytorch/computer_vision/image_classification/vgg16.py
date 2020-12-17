'''
"Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2015):
https://arxiv.org/pdf/1409.1556.pdf
'''
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Softmax
from torch.nn.init import xavier_normal_, normal_, constant_


class VGG16(Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Feature Extractor
        self.extractor = Sequential(
            Conv2d(in_channels, 64, kernel_size=3, padding=1), ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(64, 128, kernel_size=3, padding=1), ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(128, 256, kernel_size=3, padding=1), ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1), ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(256, 512, kernel_size=3, padding=1), ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
            Conv2d(512, 512, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten()
        )
        
        # Classifier
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096), ReLU(),
            Dropout(),
            Linear(4096, 4096), ReLU(),
            Dropout(),
            Linear(4096, num_classes), Softmax()
        )
        
        # Weight Initialization
        self._initialize_weights(self.extractor, self.classifier)
        
    def _initialize_weights(self, *containers):
        for modules in containers:
            for module in modules.modules():
                if isinstance(module, Conv2d):
                    xavier_normal_(module.weight)
                    constant_(module.bias, 0)
                elif isinstance(module, Linear):
                    normal_(module.weight, std=0.01)
                    constant_(module.bias, 0)

    def forward(self, x):
        ext = self.extractor(x)
        cls = self.classifier(ext)
        return cls
