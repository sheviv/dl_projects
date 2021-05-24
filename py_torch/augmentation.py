# Image Augmentation using PyTorch and Albumentations

import torch
import torchvision.transforms as transforms  # для увеличения и преобразования изображений с помощью PyTorch
import glob  # поможет нам составить список всех изображений в наборе данных
import matplotlib.pyplot as plt  # для построения изображений
import numpy as np
import torchvision
import time
import albumentations as A
from torch.utils.data import DataLoader, Dataset  # для создания пользовательского класса набора данных изображений и итеративных загрузчиков данных
from PIL import Image  # для преобразования изображения в формат RGB

# Составление списка всех изображений
image_list = glob.glob('256_ObjectCategories/*/*.jpg')

# Определение преобразований PyTorch
transform = transforms.Compose([
     transforms.ToPILImage(),  # конвертируем изображение в формат PIL
     transforms.Resize((300, 300)),
     transforms.CenterCrop((100, 100)),
     transforms.RandomCrop((80, 80)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),  # преобразование изображений в тензоры
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # нормализация изображений
     ])

# PyTorch преобразует класс набора данных и загрузчик данных
# подготовка набора данных и загрузчика данных,для использования преобразований PyTorch и аугментации изображений
class PyTorchImageDataset(Dataset):
    def __init__(self, image_list, transforms=None):
        self.image_list = image_list
        self.transforms = transforms

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        # читаем изображение из списка на основе значения индекса
        image = plt.imread(self.image_list[i])
        # PIL Image конвертирует изображение в трехканальный формат RGB
        image = Image.fromarray(image).convert('RGB')
        # преобразование изображений в массив NumPy и тип данных uint8
        image = np.asarray(image).astype(np.uint8)
        if self.transforms is not None:
            image = self.transforms(image)

        return torch.tensor(image, dtype=torch.float)

# Инициализация класса набора данных и подготовка загрузчика данных
pytorch_dataset = PyTorchImageDataset(image_list=image_list, transforms=transform)  # преобразования
pytorch_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=16, shuffle=True)  # загрузчик данных с batch=16

# Визуализация одного пакета изображений
def show_img(img):
    plt.figure(figsize=(18,15))
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)  # отсечение значений
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# извлечь один пакет изображений и вызвать show_img()
data = iter(pytorch_dataloader)
images = data.next()
show_img(torchvision.utils.make_grid(images))


# Время, затраченное на общее увеличение набора данных
start = time.time()
for i, data in enumerate(pytorch_dataloader):
    images = data
end = time.time()
time_spent = (end-start)/60
print(f"{time_spent:.3} minutes")


# Использование библиотеки Albumentations для увеличения изображения
class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.aug = A.Compose({
            A.Resize(200, 300),
            A.CenterCrop(100, 100),
            A.RandomCrop(80, 80),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-90, 90)),
            A.VerticalFlip(p=0.5),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        })

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image = plt.imread(self.image_list[i])
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))['image']  # применяем дополнения к изображению
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image, dtype=torch.float)

# Инициализация класса набора данных и подготовка загрузчика данных
alb_dataset = AlbumentationImageDataset(image_list=image_list)
alb_dataloader = DataLoader(dataset=alb_dataset, batch_size=16, shuffle=True)

# Визуализация изображений
show_img(torchvision.utils.make_grid(images))  # повернутые, обрезанные и перевернутые изображения

# кол-во времени для применения дополнений ко всему набору данных
start = time.time()
for i, data in enumerate(alb_dataloader):
    images = data
end = time.time()
time_spent = (end-start)/60
print(f"{time_spent:.3} minutes")

