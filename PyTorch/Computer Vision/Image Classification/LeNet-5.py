'''
"Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1989):
https://pdfs.semanticscholar.org/62d7/9ced441a6c78dfd161fb472c5769791192f6.pdf
'''
from torch.nn import Module, Sequential, Conv2d, Tanh, MaxPool2d, Flatten, Linear


class LeNet5(Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # Feature Extractor
        self.extractor = Sequential(
            Conv2d(in_channels, 6, kernel_size=5), Tanh(),
            MaxPool2d(2),
            Conv2d(6, 16, kernel_size=5), Tanh(),
            MaxPool2d(2), Flatten()
        )
        
        # Classifier
        self.classifier = Sequential(
            Linear(256, 120), Tanh(),
            Linear(120, 84), Tanh(),
            Linear(84, num_classes), Tanh()
        )
        
    def forward(self, x):
        ext = self.extractor(x)
        cls = self.classifier(ext)
        return cls
