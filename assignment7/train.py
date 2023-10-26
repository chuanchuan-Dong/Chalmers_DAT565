import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transform
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from letnet import Model_LeNet

os.chdir(sys.path[0])

class dense_model(nn.Module):
    def __init__(self, a = 64, b = 64):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, a),
            nn.ReLU(),
            nn.Linear(a,b),
            nn.ReLU(),
            nn.Linear(b, 10)
        )
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.sequential(x)
        return x
  
class mnist_dataset(Dataset):
      def __init__(self) -> None:
           super().__init__()
           self.transform = transform.Compose([
               transform.PILToTensor()
           ])
           self.train = MNIST(root = "./data", train=True, download=False, transform=self.transform)
           self.test = MNIST(root = "./data", train=False, download=False, transform=self.transform)

# def train_script(model, dataset):
    

  
# model = dense_model()
# summary(model, (1, 1, 28, 28))

def show_image(dataset):
    pic = dataset.train[0][0].permute(1,2,0).squeeze().numpy()
    print(pic.shape)
    # print(type(pic))
    pic = Image.fromarray(pic)
    pic.save('./pic.png')


def train_script(model:nn.Module, epochs = 10, learning_rate = 0.001):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = mnist_dataset()
    model = model.to(device)
    train_dataloader = DataLoader(dataset=dataset.train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset.test, batch_size=256, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=0.001)
    
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        number = 0
        for x, y in train_dataloader:
            _x, _y = x.float().to(device), y.to(device)
            pred_y = model(_x)
            opt.zero_grad()
            loss = criterion(pred_y, _y)
            loss.backward()
            opt.step()
            pred_y = torch.argmax(pred_y, dim=1)
            number += (pred_y == _y).sum()
            # print(loss)
        training_accuracy = float(number / len(dataset.train) * 100)
        # print(training_accuracy)
        # print(number)
        # print(loss)
        train_acc.append(training_accuracy)
        model.eval()
        number = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                _x, _y = x.float().to(device), y.to(device)
                pred_y = model(_x)
                pred_y = torch.argmax(pred_y, dim=1)
                number += (pred_y == _y).sum()
        val_accuracy = float(number / len(dataset.test) * 100)
        val_acc.append(val_accuracy)
        # break
    x_axis = [i+1 for i in range(epochs)]
    torch.save(model.state_dict(), f'./conv_model_{learning_rate}')
    plt.plot(x_axis, train_acc, label='training accuracy')
    plt.plot(x_axis, val_acc, label='validation accuracy')
    plt.legend()
    plt.show()
    
def weight_decay_train(model:nn.Module, epochs = 40, learning_rate = 0.001, decay_item = 0.001):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = mnist_dataset()
    model = model.to(device)
    train_dataloader = DataLoader(dataset=dataset.train, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset.test, batch_size=256, shuffle=True)
    

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = decay_item)
    
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        number = 0
        for x, y in train_dataloader:
            _x, _y = x.float().to(device), y.to(device)
            pred_y = model(_x)
            opt.zero_grad()
            loss = criterion(pred_y, _y)
            loss.backward()
            opt.step()
            pred_y = torch.argmax(pred_y, dim=1)
            number += (pred_y == _y).sum()
            # print(loss)
        training_accuracy = float(number / len(dataset.train) * 100)
        # print(training_accuracy)
        # print(number)
        # print(loss)
        train_acc.append(training_accuracy)
        model.eval()
        number = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                _x, _y = x.float().to(device), y.to(device)
                pred_y = model(_x)
                pred_y = torch.argmax(pred_y, dim=1)
                number += (pred_y == _y).sum()
        val_accuracy = float(number / len(dataset.test) * 100)
        val_acc.append(val_accuracy)
    # break
    x_axis = [i+1 for i in range(epochs)]
    torch.save(model.state_dict(), f'./dense_model_{learning_rate}_{decay_item}')
    plt.plot(x_axis, val_acc, label=f'{decay_item} validation accuracy')


def weight_decay_train_script(model):
    decay = [0.001 * (0.1 ** i) for i in range(5)]
    for decay_item in decay:
        print(f"start training {decay_item}")
        weight_decay_train(model=model, decay_item=decay_item)
    plt.title('Performance of different models')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    model = dense_model(a=500, b=300)
    # model = Model_LeNet(1)
    weight_decay_train(model)
    # summary(model, (1,1,28,28))
    # print(model(torch.rand([256,28,28])))
    # weight_decay_train_script(model)
    # train_script(model,20)