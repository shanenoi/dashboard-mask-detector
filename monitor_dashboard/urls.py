from django.contrib import admin
from django.http import StreamingHttpResponse
from django.urls import path
from monitor_dashboard import MASK_MODEL
from monitor_dashboard.camera import ( VideoCamera,
                                       StreamVideo )

urlpatterns = [
    path(
        'monitor/<int:camera_id>',
        lambda request, camera_id: StreamingHttpResponse(
            StreamVideo(VideoCamera(camera_id, MASK_MODEL)),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    )
]
