# Утилиты сегментации изображений
# несколько утилит и функций сегментации изображений
import torchvision.transforms as transforms
import cv2
import numpy as np
import numpy
import torch
from my_label_color_map import label_color_map as label_map  # импорт цветов

# Определить преобразования изображения
# нормализовать изображения, используя среднее значение и стандартное значение из их обучающего набора
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])


def get_segment_labels(image, model, device, detection_threshold=0.8):
    # print(f"get_segment_labels image: {image}")
    # print(f"get_segment_labels model: {model}")
    # print(f"get_segment_labels device: {device}")
    # transform the image to tensor
    image = transform(image).to(device)
    print(f"get_segment_labels image_1: {image}")
    # print(f"get_segment_labels image_1: {type(image)}")
    image = image.unsqueeze(0)  # добавляем батч измерение
    print(f"get_segment_labels image_2: {image}")
    # print(f"get_segment_labels image_2: {type(image)}")
    outputs = model(image)  # выходной словарь после того, как модель выполняет прямой проход через изображение
    print(f"get_segment_labels outputs: {outputs}")
    # print(f"get_segment_labels outputs: {type(outputs)}")

    # # get all the predicited class names
    # pred_classes = [label_map[i] for i in outputs[0]['labels'].cpu().numpy()]
    # # get score for all the predicted objects
    # pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # # get all the predicted bounding boxes
    # pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # # get boxes above the threshold score
    # boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    # return boxes, pred_classes, outputs[0]['labels']
    return outputs

# def draw_boxes(classes, labels, image):
#     # read the image with OpenCV
#     image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
#     for i, box in range(0, len(label_map)):
#         color = label_map[labels[i]]
#         cv2.rectangle(
#             image,
#             (int(box[0]), int(box[1])),
#             (int(box[2]), int(box[3])),
#             color, 2
#         )
#         cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
#                     lineType=cv2.LINE_AA)
#     return image


# применить цветовые маски в соответствии со значениями тензора в выходном словаре get_segment_labels()
def draw_segmentation_map(outputs):
    # print(f"draw_segmentation_map outputs: {outputs}")
    # labels = torch.argmax(torch.tensor(outputs).squeeze(), dim=0).detach().cpu().numpy()
    labels = torch.argmax(outputs.detach().clone().squeeze(), dim=0).cpu().numpy()

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


    # seg = [i for i, k in segmented_image.items()]
    # print(f"seg: {seg}")
    # return seg


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
