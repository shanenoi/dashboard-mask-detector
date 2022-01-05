import cv2

class VideoCamera(object):
    def __init__(self, camera_id):
        self.video = cv2.VideoCapture(camera_id)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        return image
