import multiprocessing
import numpy as np
import torch
import torchvision.models
from multiprocessing import Process, shared_memory

import cv2

# img = decode_image("H:/Projects/University/NeauralNetworks/Datasets/grace_hopper.webp")
from PIL import Image
# img = Image.open("H:/Projects/University/NeauralNetworks/Datasets/grace_hopper.webp")

# Step 1: Initialize model with the best available weights
weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()


bbox_info_memory = None

# Linux
frame = None
# Windows
image_array = None


# Create array with underlying buffer of bbox_memory
# Create array with underlying buffer of image_memory. Calculate buffer size. You can create the shared mem and corresponding array at the time of first image coming
# Convert image into array, calculate size and create shared memory
# See https://docs.python.org/3/library/multiprocessing.shared_memory.html 
# Then manipulate shared memory as with arrays

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

def detect_object(img):
    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    top_pred = torch.split(prediction["boxes"], split_size_or_sections=1)[0]
    return top_pred[0], prediction["labels"][0]

def detect_objects(bbox_info_memory):
    while True:
        # read frame
        if image_array is None:
            continue
        image = np.array(image_array)
        if image is None:
            continue
        im_pil = Image.fromarray(image)
        boxes, label_idx = detect_object(im_pil)
        img_boxes = boxes.to(torch.int64).tolist()
        # Copy bounding box props
        bbox_info_memory[0] = img_boxes[0]
        bbox_info_memory[1] = img_boxes[1]
        bbox_info_memory[2] = img_boxes[2]
        bbox_info_memory[3] = img_boxes[3]
        bbox_info_memory[4] = label_idx


cv2.namedWindow("Preview")
vc = cv2.VideoCapture(0)
object_detection_process = None

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
i = 0
while rval:
    cv2.imshow("Preview", frame)
    rval, frame = vc.read()
    im_pil = Image.fromarray(frame)
    if image_array is None:
        image_array = multiprocessing.Array('i', len(frame))
    if bbox_info_memory is None:
        bbox_info_memory = multiprocessing.Array('i', 5)
    if object_detection_process is None:
        object_detection_process = Process(target=detect_objects, args=(bbox_info_memory,))
        object_detection_process.start()

    # WINDOWS
    image_array = frame

    if bbox_info_memory[0] != -1:
        cv2.rectangle(frame, (bbox_info_memory[0], bbox_info_memory[1]), (bbox_info_memory[2], bbox_info_memory[3]), (0, 0, 255), 2)
        bbox_info_memory[0] = -1

    # boxes, labels = detect_object(im_pil)
    # img_boxes = boxes.to(torch.int64).tolist()
    # for box, label in zip(img_boxes, labels):
    #     print(label)
    #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,  0, 255), 2)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        object_detection_process.kill()
        break

cv2.destroyWindow("Preview")
vc.release()
