# Утилиты сегментации изображений
# несколько утилит и функций сегментации изображений


import torchvision.transforms as transforms
import cv2
import numpy as np
import numpy
import torch
from label_color_map import label_color_map as label_map  # импорт цветов

# Определить преобразования изображения
# нормализовать изображения, используя среднее значение и стандартное значение из их обучающего набора
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#
def get_segment_labels(image, model, device):
    image = transform(image).to(device)
    image = image.unsqueeze(0)  # добавляем батч измерение
    outputs = model(image)  # выходной словарь после того, как модель выполняет прямой проход через изображение
    # print(type(outputs))
    # print(outputs['out'].shape)
    # print(outputs)
    return outputs


# применить цветовые маски в соответствии со значениями тензора в выходном словаре get_segment_labels()
def draw_segmentation_map(outputs):
    # squeeze() к выходам после преобразования их в массив NumPy,
    # загрузка тензоров в ЦП, получаем позиции индекса(torch.argmax())
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    # три массива NumPy для карт красного, зеленого и синего цветов и заполнить нулями.
    # Размер аналогичен размеру меток, которые в labels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    # цикл for 21 раз(количество рассматриваемых меток)
    for label_num in range(0, len(label_map)):
        # используя индексную переменную(применяя красный, зеленый и синий цвета к массивам NumPy)
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
    # скложить последовательность цветовой маски по новой оси(окончательное сегментированное изображение цветовой маски)
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    # возвращаем сегментированную маску
    return segmented_image


# применить сегментированные цветовые маски поверх исходного изображения
def image_overlay(image, segmented_image):
    alpha = 0.6
    beta = 1 - alpha
    gamma = 0
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    # применить маску segmented_image поверх изображения
    # alpha - управление прозрачностью изображения
    # beta - вес, примененный к исходному изображению
    # gamma - скаляр(добавляется к каждой сумме)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image