# Трансферное обучение - перенос обучения
# VGG16 для классификации изображений CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models


# если есть GPU, он будет выбран, иначе CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Скачивание и подготовка набора данных
# операции предварительной обработки изображений
transform = transforms.Compose(
    [transforms.Resize((224, 224)),  # изменяет размер изображения до размера 224 × 224(VGG принимает входное изображение 224 × 224)
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)


# Загрузка сети VGG16
vgg16 = models.vgg16(pretrained=True)  # pretrained=True - загрузка весов ImageNet для предварительно обученной модели
vgg16.to(device)  # загружает модель на устройство(процессор или графический процессор)


# Замораживание сверточных весов
# vgg16 - 1000 классов, а нужно 10
# замораживание весов Conv2d() заставит модель использовать все предварительно обученные веса
# изменить количество классов
vgg16.classifier[6].out_features = 10  # классификатор VGG-16 из 6-слойного массива
# замораживание весов Conv2d()
for param in vgg16.features.parameters():
    param.requires_grad = False


# Оптимизатор и функция потерь
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# Функции обучения и проверки
def validate(model, test_dataloader):
    # вычисление потери и точности
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(test_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        # не распространять градиенты в обратном направлении(только во время обучения)
        loss = criterion(output, target)
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    val_loss = val_running_loss / len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct / len(test_dataloader.dataset)
    return val_loss, val_accuracy


def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()  # вычислить градиенты и выполнить обратное распространение
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    return train_loss, train_accuracy


# 10 эпо(для каждой эпохи вызываются методы fit() и validate())
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_accuracy = fit(vgg16, trainloader)
    val_epoch_loss, val_epoch_accuracy = validate(vgg16, testloader)
    # После каждой эпохи сохранять значения точности обучения и потерь
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
print((end-start)/60, 'minutes')


# Визуализация графиков
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()