# Применение сегментации к изображениям

import torchvision
import numpy
import torch
import argparse
import segmentation_utils
import cv2
from PIL import Image
import gc

torch.cuda.empty_cache()

# Синтаксический анализатор аргументов, инициализировать модель и устройство
# один аргумент - путь к изображению
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
args = vars(parser.parse_args())

# инициализация модели и вычислительного устройства
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# перевести модель в режим оценки и загрузить на вычислительное устройство
# torch.cuda.empty_cache()
model.eval().to(device)


# Чтение изображения и применить сегментацию к нему
image = Image.open(args['input'])
# сделать прямой проход и получить выходной словарь
outputs = segmentation_utils.get_segment_labels(image, model, device)
# получить данные из ключа `out`
outputs = outputs['out']
# применить цветовые маски для различных классов, обнаруженных на изображении
segmented_image = segmentation_utils.draw_segmentation_map(outputs)


# Наложить сегментированную маску и сохранение
final_image = segmentation_utils.image_overlay(image, segmented_image)
# сохраним окончательный результат
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# показать сегментированное изображение и сохранить на диск
cv2.imshow('Segmented image', final_image)
cv2.waitKey(0)
cv2.imwrite(f"outputs/{save_name}.jpg", final_image)
