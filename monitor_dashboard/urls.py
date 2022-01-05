from django.contrib import admin
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.urls import path
from monitor_dashboard import LIST_CAMERAS, FACE_NET, MASK_NET
from monitor_dashboard.detector import Detector

import cv2


def StreamVideo(camera):
    while True:
        _, frame = cv2.imencode('.jpg', camera.get_frame())
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')


urlpatterns = [
    path('monitor/<int:camera_id>',
         lambda request, camera_id: StreamingHttpResponse(
             StreamVideo(Detector(camera_id, FACE_NET, MASK_NET)),
             content_type='multipart/x-mixed-replace; boundary=frame'
         )),
    path('', lambda request: render(request, "index.html", { 'list_cameras': LIST_CAMERAS }))
]
