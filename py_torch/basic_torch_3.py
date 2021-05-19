# Fashion MNIST Classification using PyTorch

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Определение некоторых констант
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# Определение преобразований
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Скачать данные
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True,
                                             transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True,
                                            transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=True)

# изображения вместе с подписями(нужны названия вещей - классы)
classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal',
           'Shirt','Sneaker','Bag','Ankle Boot')
# визуализировать картинки
for batch_1 in trainloader:
    batch = batch_1
    break
print(batch[0].shape)  # as batch[0] содержит пиксели изображения -> тензоры
print(batch[1].shape)  # batch[1] содержит метки -> тензоры
plt.figure(figsize=(12, 8))
for i in range(batch[0].shape[0]):
    plt.subplot(4, 8, i+1)
    plt.axis('off')
    plt.imshow(batch[0][i].reshape(28, 28), cmap='gray')
    plt.title(classes[batch[1][i]])
    plt.savefig('fashion_mnist.png')
plt.show()


# Создать архитектуру LeNet CNN
# модуль для построения архитектуры LeNet CNN(nn.Module от PyTorch для определения слоев)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=256, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = LeNet()


# Функция потерь и оптимизатор
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)


# Обучение
# если есть GPU, он будет выбран, иначе CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# если есть CUDA, то он будет перенесен на устройство CUDA
net.to(device)

# точность обучения на обучающих данных и точность проверки на тестовых данных
def calc_acc(loader):
    correct = 0
    total = 0
    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return ((100 * correct) / total)


# Обучение нейронной сети на данных
def train():
    # хранят потери, точность обучения и точность тестирования для каждой из 10 эпох
    epoch_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # установить градиенты параметров на ноль
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss.append(running_loss / 15000)
        train_acc.append(calc_acc(trainloader))
        test_acc.append(calc_acc(testloader))
        print('Epoch: %d of %d, Train Acc: %0.3f, Test Acc: %0.3f, Loss: %0.3f'
              % (epoch + 1, NUM_EPOCHS, train_acc[epoch], test_acc[epoch], running_loss / 15000))

    return epoch_loss, train_acc, test_acc

start = time.time()
epoch_loss, train_acc, test_acc = train()
end = time.time()
print('%0.2f minutes' % ((end - start) / 60))


# Визуализация линейных графиков
# Loss Plot
plt.figure()
plt.plot(epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('fashion_loss.png')
plt.show()
# Training Accuracy
plt.figure()
plt.plot(train_acc)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.savefig('fashion_train_acc.png')
plt.show()
# Test Accuracy
plt.figure()
plt.plot(test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.savefig('fashion_test_acc.png')
plt.show()