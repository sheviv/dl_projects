# python3 streaming.py
from mss import mss
import numpy as np
import time
import torch
import torchvision
import cv2
import argparse
from PIL import Image
from my_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from my_instance_util import get_stream

# cv2.putText() method
# font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# org
org = (0, 15)
# fontScale
fontScale = 1
# Red color in BGR
color = (0, 255, 0)
# Line thickness of 2 px
thickness = 1

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

sct = mss()
bounding_box = {'top': 0, 'left': 0, 'width': 800, 'height': 640}

while True:
    """
    изменение размерности с _,_,4 на _,_,3
    """
    sct_img = sct.grab(bounding_box)
    frame_np = np.array(sct_img)
    print(f"frame_np: {type(frame_np)}??? {frame_np.shape}")
    frame_four = torch.from_numpy(frame_np)
    frame = frame_four[:, :len(frame_four[0]) - 1]
    # print(f"frame: {frame}")
    # orig_image = frame.copy()
    orig_image = frame.clone()
    print(f"orig_image: {orig_image.shape}")
    with torch.no_grad():
        print(f"frame__1: {type(frame)}")
        print(f"frame__2: {frame.shape}")
        # print(f"frame_[-1]: {frame[-1]}")

        # im = Image.fromarray(np.uint8(frame))
        # image = get_stream(frame, model, device)  # common
        image = get_stream(frame, model, device)
        print(f"image_1_stream: {image}")
        print(f"image_1_stream: {type(image)}")
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        print(f"image_2_stream: {image}")
        print(f"image_2_stream: {type(image)}")
        masks, boxes, labels = get_outputs(image, model, 0.965)

    result = draw_segmentation_map(orig_image, masks, boxes, labels)

    cv2.imshow('stream', result)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
