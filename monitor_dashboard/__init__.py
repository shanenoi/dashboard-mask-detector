from tensorflow.keras.models import load_model

import cv2
import os
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
FACE_NET = cv2.dnn.readNet(os.path.join("models", "deploy.prototxt"), os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel"))
MASK_NET = load_model(os.path.join("models", "mask_detector.model"))
