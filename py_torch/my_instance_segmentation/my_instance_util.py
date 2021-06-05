import torchvision.transforms as transforms
import cv2
import numpy as np
import numpy
import torch
from PIL import Image

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])


def get_segment_labels(image, model, device):
    # keep a copy of the original image for OpenCV functions and applying masks
    # transform the image
    image = transform(image).to(device)  # common
    # print(f"get_segment_labels image_1: {image}")
    # add a batch dimension
    image = image.unsqueeze(0)
    # print(f"get_segment_labels image_2: {image}")
    # print(f"get_segment_labels outputs: {type(model(image))}")
    outputs = model(image)
    return outputs


def get_stream(image, model, device):
    image = transform(image).to(device)  # common
    # print(f"get_segment_labels image_1: {image}")
    image = image.unsqueeze(0)
    # print(f"get_segment_labels image_2: {image}")
    # print(f"get_segment_labels outputs: {type(model(image))}")
    outputs = model(image)
    return outputs
