# Digit MNIST Classification using PyTorch

import torch
import torchvision
import \
    torchvision.transforms as transforms  # преобразовании значения пикселей изображений(нормализациея и стандартизация)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import datasets

# Определение преобразований
transform = transforms.Compose(
    [transforms.ToTensor(),  # конвертируем набор данных в тензоры
     transforms.Normalize((0.5,), (0.5,))]  # нормализация набора данных
)
# для RGB Нормализовать ((0,5, 0,5, 0,5), (0,5, 0,5, 0,5))


# Скачать данные
trainset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform  # напрямую преобразует данные в тензоры
)
trainloader = torch.utils.data.DataLoader(  # преобразования данных в batch
    trainset,
    batch_size=4,
    shuffle=True
)
testset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(  # преобразования данных в batch
    testset,
    batch_size=4,
    shuffle=False
)


# Визуализация изображений
# for batch_1 in trainloader:  # посмотреть первый батч из 4 картинок
#     batch = batch_1
#     break
# print(batch[0].shape)  # batch[0] содержит пиксели изображения -> тензоры
# print(batch[1])  # batch[1] содержит метки -> тензоры
# plt.figure(figsize=(12, 8))
# for i in range(batch[0].shape[0]):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(batch[0][i].reshape(28, 28), cmap='gray')
#     plt.title(int(batch[1][i]))
#     plt.savefig('digit_mnist.png')
# plt.show()


# Определить нейронную сеть
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20,
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
                               kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# если есть GPU, он будет выбран, иначе CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# если есть CUDA, то он будет перенесен на устройство CUDA
net = Net().to(device)


# Оптимизатор и функция потерь
# функция потерь
criterion = nn.CrossEntropyLoss()
# Оптимизатор
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Обучение нейронной сети на данных
# 15000 пакетов по 4 изображения
def train(net):
    start = time.time()  # засечь время обучения модели
    for epoch in range(10):  # 10 эпох
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # пиксели данных и метки для графического процессора, если они доступны
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # установить градиенты параметра на ноль
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # распространяем фун-ию ошибок
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5000 == 4999:  # каждые 5000 мини-партий
                print('[Epoch %d, %5d Mini Batches] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5000))
                running_loss = 0.0
    end = time.time()
    print('Done Training')
    print('%0.2f minutes' % ((end - start) / 60))
# train(net)


# Тестирование нашей сети на тестовом наборе
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on test images: %0.3f %%' % (
    100 * correct / total))
