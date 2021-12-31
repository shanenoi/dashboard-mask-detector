import cv2
import torch
import warnings

def list_cam():
    position = 0
    while True:
        cap = cv2.VideoCapture(position)
        if not cap.read()[0]:
            break
        else:
            yield position
        cap.release()
        position += 1


LIST_CAMERAS = list(list_cam())
warnings.filterwarnings("ignore")
MASK_MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path='./mask_yolov5m.pt', force_reload=True)
