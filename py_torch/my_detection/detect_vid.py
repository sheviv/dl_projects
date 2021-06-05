import torchvision
import cv2
import torch
import argparse
import time
import detect_utils

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-d', '--id', default=0,
                    help='path to the input data')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())
args["id"] = int(args["id"])
# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=args['min_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# video_capture = cv2.VideoCapture(args['input'])  # in globals()
# if video_capture.isOpened() == False:
#     print('Error while trying to read video. Please check path again')
# # get the frame width and height
# frame_width = int(video_capture.get(3))
# frame_height = int(video_capture.get(4))
# save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
# # define codec and create VideoWriter object
# out = cv2.VideoWriter(f"outputs/{save_name}.mp4",
#                       cv2.VideoWriter_fourcc(*'mp4v'), 30,
#                       (frame_width, frame_height))
video_capture = cv2.VideoCapture(args['id'])  # in globals()
if not video_capture.isOpened():
    print('Error while trying to read video. Please check path again')

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second
while video_capture.isOpened():
    ret, frame = video_capture.read()
    # orig_image = frame.copy()
    if ret:
        start_time = time.time()
        # load the model onto the computation device
        with torch.no_grad():
            model = model.eval().to(device)  # ?????? model.to(device).eval()
            # get predictions for the current frame
            boxes, classes, labels = detect_utils.predict(frame, model, device, 0.8)
        # draw boxes and show current frame on screen
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # press `q` to exit
        wait_time = max(1, int(fps / 4))
        cv2.imshow('Video', image)
        # if args['input']:
        #     out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
video_capture.release()
# close all frames and video windows
cv2.destroyAllWindows()
