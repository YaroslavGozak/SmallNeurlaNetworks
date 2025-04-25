# Below is a version of the multithreaded YOLOv5n demo tailored for a Raspberry Pi (e.g. Pi 4/Zero 2 W) using the Pi Camera (via Picamera2) and CPU-only PyTorch. Key changes:
# - Picamera2 for low-latency capture at 640×480
# - TorchScript compilation of the model for faster startup
# - Limit PyTorch threads to avoid over-saturating the small CPU
# - Drop to a single-frame queue to always process the freshest image

# Tips for Raspberry Pi
# Install PyTorch: Use the official Pi wheels (e.g. via pip install torch-<version>-cp38-none-linux_armv7l.whl)
# Dependencies:

# `bash`
# sudo apt update
# sudo apt install libatlas-base-dev libopenblas-dev libomp-dev libjpeg-dev
# pip install opencv-python torch torchvision picamera2
# 
# Resolution & FPS: 640×480 at ~10–15 FPS on Pi 4; drop to 320×240 for higher speed
# Batching: Always process the latest frame (queue size=1) to keep display smooth
# Thread tuning: Adjust torch.set_num_threads() to match your Pi’s cores

import cv2
import torch
import threading
import queue
from picamera2 import Picamera2

# ─── 0. PyTorch/OS tweaks ───────────────────────────────────────────────────────
torch.set_num_threads(2)   # limit CPU threads
torch.backends.mkldnn.enabled = False

# ─── 1. Load & optimize YOLOv5n ────────────────────────────────────────────────
# (ensure you have installed torch and ultralytics/yolov5 repo on the Pi)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)
model.conf = 0.25
model.to('cpu').eval()
# compile to TorchScript for faster inference
ts_model = torch.jit.script(model)

# ─── 2. Set up Picamera2 capture ───────────────────────────────────────────────
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()

# ─── 3. Shared data structures ─────────────────────────────────────────────────
frame_queue = queue.Queue(maxsize=1)
results = {'boxes': None, 'scores': None, 'labels': None}
lock    = threading.Lock()
running = True

# ─── 4. Detection worker ───────────────────────────────────────────────────────
def detection_worker():
    global running
    while running:
        try:
            img = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # convert BGR to RGB and to tensor
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # batch dim

        # inference
        with torch.no_grad():
            pred = ts_model(tensor)[0]

        det = pred.cpu().numpy()  # Nx6: x1,y1,x2,y2,conf,cls
        with lock:
            if det.shape[0]:
                results['boxes']  = det[:, :4]
                results['scores'] = det[:, 4]
                results['labels'] = det[:, 5].astype(int)
            else:
                results['boxes'] = results['scores'] = results['labels'] = None

# start the thread
t = threading.Thread(target=detection_worker, daemon=True)
t.start()

# ─── 5. Main loop: capture & display ────────────────────────────────────────────
try:
    while True:
        # capture frame as BGR numpy array
        frame = picam2.capture_array()

        # queue for detection (drop old if busy)
        if not frame_queue.full():
            frame_queue.put(frame.copy())

        # overlay last detections
        with lock:
            boxes, scores, labels = results['boxes'], results['scores'], results['labels']

        if boxes is not None:
            for (x1,y1,x2,y2), conf, cls in zip(boxes, scores, labels):
                x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
                name = model.names[cls]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f'{name} {conf:.2f}',
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow('Pi YOLOv5n', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # cleanup
    running = False
    t.join()
    picam2.stop()
    cv2.destroyAllWindows()
