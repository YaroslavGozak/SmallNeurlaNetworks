import cv2
import torch
import threading
import queue

import numpy as np
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

# ─── 1. Load YOLOv5n model ──────────────────────────────────────────────────────
# Make sure you have 'ultralytics' YOLOv5 repo available for torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.25  # confidence threshold
model.to('cpu').eval()

# ─── 2. Shared data structures ─────────────────────────────────────────────────
frame_queue = queue.Queue(maxsize=1)  # keep only the latest frame
results = {'boxes': None, 'scores': None, 'labels': None}
lock    = threading.Lock()
running = True

# ─── 3. Detection worker thread ────────────────────────────────────────────────
def detection_worker():
    global running
    while running:
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Run inference
        preds = model(frame)

        # Parse predictions (tensor Nx6: x1, y1, x2, y2, conf, cls)
        det = preds.xyxy[0].cpu().numpy()
        with lock:
            if det.size:
                results['boxes']  = det[:, :4]
                results['scores'] = det[:, 4]
                results['labels'] = det[:, 5].astype(int)
            else:
                results['boxes'] = results['scores'] = results['labels'] = None

# Start worker
thread = threading.Thread(target=detection_worker, daemon=True)
thread.start()

# ─── 4. Main capture & display loop ───────────────────────────────────────────
cap = cv2.VideoCapture(0)  # or path to video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Send the latest frame to the detection thread (drop if busy)
    if not frame_queue.full():
        frame_queue.put(frame.copy())

    # Overlay last detections
    with lock:
        boxes, scores, labels = results['boxes'], results['scores'], results['labels']

    if boxes is not None:
        for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            name = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} {conf:.2f}',
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('YOLOv5n Edge Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── 5. Cleanup ────────────────────────────────────────────────────────────────
running = False
thread.join()
cap.release()
cv2.destroyAllWindows()