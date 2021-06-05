import time
import torch
import torchvision
import cv2
import argparse
from PIL import Image
from my_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from my_instance_util import get_segment_labels


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


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', required=True,
                    help='path to the input data')
parser.add_argument('-t', '--threshold', default=0.965, type=float,
                    help='score threshold for discarding detection')
args = vars(parser.parse_args())
args["id"] = int(args["id"])

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model on to the computation device and set to eval mode
model.to(device).eval()

video_capture = cv2.VideoCapture(args['id'])

if not video_capture.isOpened():
    print('Error while trying to read video. Please check path again')

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second

while video_capture.isOpened():
    ret, frame = video_capture.read()
    # print(f"frame_1: {type(frame)}")
    orig_image = frame.copy()
    print(f"orig_image: {orig_image.shape}")
    if ret:
        start_time = time.time()
        with torch.no_grad():
            # print(f"frame__1: {type(frame)}")
            # print(f"frame__2: {frame.shape}")
            # print(f"frame_[-1]: {frame[-1]}")
            image = get_segment_labels(frame, model, device)
            # print(f"image_1_stream: {image}")
            # print(f"image_1_stream: {type(image)}")
            # add a batch dimension
            image = image.unsqueeze(0).to(device)
            # print(f"image_2_stream: {image}")
            # print(f"image_2_stream: {type(image)}")
            masks, boxes, labels = get_outputs(image, model, args['threshold'])

        result = draw_segmentation_map(orig_image, masks, boxes, labels)
        # get the end time
        end_time = time.time()
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add current fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # put the FPS text on the current frame
        # cv2.putText(final_image, f"{fps:.3f} FPS", (20, 35),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"{fps:.3f} FPS", org, font, fontScale, color, thickness)
        # Display the resulting frame
        print(f"result: {type(result)}")
        cv2.imshow('Video', result)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
