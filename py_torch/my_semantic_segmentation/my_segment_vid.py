import argparse
import cv2
import my_segment_util
import time
import torch
import torchvision

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
parser.add_argument('-i', '--id', help='id camera')
# parser.add_argument('-f', '--folder', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,
                    help='minimum input size for the FasterRCNN network')
# parser.add_argument('-o', '--output', type=pathlib.Path, metavar="<path>",
#                     help="Path for writing output video file.")
args = vars(parser.parse_args())
args["id"] = int(args["id"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=args['min_size'])
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# models for segmentation
model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
# model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
# model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model = model.eval().to(device)

video_capture = cv2.VideoCapture(args['id'])

if (video_capture.isOpened() == False):
    print('Error while trying to read video. Please check path again')

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        start_time = time.time()
        with torch.no_grad():
            outputs = my_segment_util.get_segment_labels(frame, model, device)

        # отрисовка поля и показ текущего кадра
        segmented_image = my_segment_util.draw_segmentation_map(outputs['out'])
        # segmented_image = my_segment_util.draw_segmentation_map(outputs)
        final_image = my_segment_util.image_overlay(frame, segmented_image)

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
        cv2.putText(final_image, f"{fps:.3f} FPS", org, font, fontScale, color, thickness)
        # Display the resulting frame
        cv2.imshow('Video', final_image)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# # load the model onto the computation device
# with torch.no_grad():
#     # model = model.eval().to(device)
#     # while (video_capture.isOpened()):
#     if video_capture.isOpened():
#         while True:
#             start_time = time.time()
#             # Capture frame-by-frame
#             ret, frame = video_capture.read()
#             print(f"FPS: {1.0 / (time.time() - start_time):.3f}")
#
#             frame = cv2.putText(frame, f"FPS: {(1.0 / (time.time() - start_time)) / 1000:.3f}",
#                                 org, font, fontScale, color, thickness)
#             if ret == True:
#                 # # get the start time
#                 with torch.no_grad():
#                     # get predictions for the current frame
#                     outputs = my_segment_util.get_segment_labels(frame, model, device)
#
#                 # отрисовка поля и показ текущего кадра
#                 segmented_image = my_segment_util.draw_segmentation_map(outputs['out'])
#                 # segmented_image = my_segment_util.draw_segmentation_map(outputs)
#                 final_image = my_segment_util.image_overlay(frame, segmented_image)
#
#                 # Display the resulting frame
#                 cv2.imshow('Video', final_image)
#                 # press `q` to exit
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             else:
#                 break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()
