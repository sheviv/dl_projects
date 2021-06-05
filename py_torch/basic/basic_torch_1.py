# torch.nn.Module - является базовым классом для всех модулей нейронной сети
# torch.nn.Functional -  все промежуточные операции во время прямого прохода сети

# Определение:
# оптимизатора
# функций потерь
# обратного распространения

import torch
import torch.nn as nn
import torch.nn.functional as F


# архитектура Lenet-5
class Net(nn.Module):  # класс Net наследует все свойства nn.Module
    def __init__(self):
        super(Net, self).__init__()  # унаследуем все свойства, если nn.Module
        self.conv1 = nn.Conv2d(1, 6, (3, 3))  # (канал входного изображения, выходные каналы, размер ядра)
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        # в линейном слое (выходные каналы из conv2d x ширина x высота)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # (канал входного изображения, выходные каналы)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)  # сглаживает линейные слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = Net()
# print(model)

# Определение оптимизатора и функции потерь
import torch.optim as optim
loss_function = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)  # параметры модели, шаг обучения

# Фиктивный ввод и обратное распространение
input = torch.rand(1, 1, 32, 32)  # фиктивный вход размером 32 × 32
out = model(input)

labels = torch.rand(10)
labels = labels.view(1, -1)  # изменить размер для той же формы, что и на выходе

# функция потерь и обратное распространение ошибки
loss = loss_function(out, labels)
loss.backward()
