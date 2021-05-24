import torchvision
import cv2
import torch
import argparse
import time
import segmentation_utils
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
args = vars(parser.parse_args())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# download or load the model from disk
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
# load the model onto the computation device
model = model.eval().to(device)
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# ширина и высота рамки
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# определить кодек(mp4) и создать объект VideoWriter для сохранения видео
out = cv2.VideoWriter(f"outputs/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))
frame_count = 0  # подсчитать общее количество кадров
total_fps = 0  # чтобы получить окончательные кадры в секунд


# Применение сегментации изображений к каждому видеокадру
# цикл while(перебор всех кадров в видео)
while (cap.isOpened()):
    # захватить каждый кадр видео
    ret, frame = cap.read()
    if ret == True:
        # узнать время начала
        start_time = time.time()
        with torch.no_grad():
            # получить прогнозы для текущего кадра
            outputs = segmentation_utils.get_segment_labels(frame, model, device)
        # отрисовка поля и показ текущего кадра
        segmented_image = segmentation_utils.draw_segmentation_map(outputs['out'])
        final_image = segmentation_utils.image_overlay(frame, segmented_image)
        # время окончания
        end_time = time.time()
        # значения fps
        fps = 1 / (end_time - start_time)
        # добавить fps к общему fps
        total_fps += fps
        # увеличивать количество кадров
        frame_count += 1
        # клавиша `q` для выхода
        wait_time = max(1, int(fps / 4))
        cv2.imshow('image', final_image)
        out.write(final_image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break


# Уничтожение всех окон видео и вычисление среднего FPS
# освобождение объекта VideoCapture(), закрытие всех окон и рассчет среднего FPS
cap.release()
cv2.destroyAllWindows()
# рассчитать и вывести средний FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
