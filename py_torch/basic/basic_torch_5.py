# Пользовательский набор данных и загрузчик данных в PyTorch

# __init__() - инициализируем все, что используется в классе
# __len__() - возвращает длину набора данных(количество выборок в наборе данных)
# __getitem__() - возвращает образец из набора данных, когда мы предоставляем ему значение индекса

# Класс для набора данных
# В фиктивном наборе данных создается массив Numpy и передается его в качестве входных данных классу.
import numpy as np
from torch.utils.data import Dataset
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        """
        возвращаем длину данных массива Numpy
        """
        return len(self.data)
    def __getitem__(self, idx):
        """
        возвращает элемент из данных, соответствующих параметру index (idx)
        """
        return self.data[idx]
sample_data = np.arange(0, 10)  # создаем простой массив Numpy из 10 элементов
print('The whole data: ', sample_data)
dataset = ExampleDataset(sample_data)  # инициализируем объект набора данных класса и передаем sample_data
print('Number of samples in the data: ', len(dataset))
print(dataset[2])
print(dataset[0:5])


# Импорт библиотек и определение вспомогательных функций
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# если есть GPU, он будет выбран, иначе CPU
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()


# Загрузка и подготовка данных
# Загрузка данны(отделить значения пикселей от меток изображений)
df_train = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')
# получение значений пикселей изображения и меток
train_labels = df_train.iloc[:, 0]
train_images = df_train.iloc[:, 1:]
test_labels = df_test.iloc[:, 0]
test_images = df_test.iloc[:, 1:]

# Определение преобразования изображения
transform = transforms.Compose(
    [transforms.ToPILImage(),  # конвертируем данные в изображение PIL
     transforms.ToTensor(),  # конвертируем в тензоры PyTorch
     transforms.Normalize((0.5, ), (0.5, ))])  # нормализуем данные изображения


# класс набора данных (MNISTDataset), набора данных и определим загрузчики данных
class MNISTDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
    def __len__(self):
        return (len(self.X))
    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1)
        if self.transforms:
            data = self.transforms(data)
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

train_data = MNISTDataset(train_images, train_labels, transform)
test_data = MNISTDataset(test_images, test_labels, transform)
# загрузка данных
trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
testloader = DataLoader(test_data, batch_size=128, shuffle=True)


# Определение нейронной сети
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
net = Net().to(device)
# print(net)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Функции обучения и проверки
def train(net, trainloader):
    for epoch in range(10):  # no. of epochs
        running_loss = 0
        for data in trainloader:
            # пиксели данных и метки для графического процессора, если они доступны
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
              (epoch + 1, running_loss / len(trainloader)))
    print('Done Training')


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: %0.3f %%' % (100 * correct / total))

train(net, trainloader)
test(net, testloader)
