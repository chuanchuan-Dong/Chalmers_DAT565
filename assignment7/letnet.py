import torch.nn.functional as F
import torch.nn as nn
import torch
from torchsummary import summary
 
class Model_LeNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes=10):
        super(Model_LeNet, self).__init__()
        self.sequential1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten()
        )
        
        self.sequential2 = nn.Sequential(
            nn.Linear(400, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
 
    def forward(self, x):
        x = self.sequential1(x)
        # print(x.shape)
        x = self.sequential2(x)
        return x

if __name__ == '__main__':
    model = Model_LeNet(in_channels=1)
    summary(model, (1,28,28))