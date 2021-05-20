"""
American Sign Language Detection
"""

'''
USAGE:
python preprocess_image.py --num-images 1200
'''
import os
import cv2
import random
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-images', default=1200, type=int,
                    help='number of images to preprocess for each category')
args = vars(parser.parse_args())
print(f"Preprocessing {args['num_images']} from each category...")

# Получить все пути к каталогам
dir_paths = os.listdir('../input/asl_alphabet_train/asl_alphabet_train')
dir_paths.sort()
root_path = '../input/asl_alphabet_train/asl_alphabet_train'


# Предварительная обработка изображений и их сохранение на диск(перебираем все пути dir_path)
for idx, dir_path in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    # получаем все изображения в соответствующем каталоге классов
    all_images = os.listdir(f"{root_path}/{dir_path}")
    # создаем папку preprocessed_image и внутри нее папку класса, которую мы в данный момент перебираем
    os.makedirs(f"../input/preprocessed_image/{dir_path}", exist_ok=True)
    # будет повторяться --num-images количество раз
    for i in range(args['num_images']):  # сколько изображений предварительно обработать для каждой категории
        rand_id = (random.randint(0, 2999))
        image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(f"../input/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", image)
