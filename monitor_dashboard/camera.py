import cv2

class VideoCamera(object):
    def __init__(self, camera_id, detect_model):
        self.detect_model = detect_model
        self.video = cv2.VideoCapture(camera_id)
    
    def __del__(self):
        self.video.release()

    def detect_target(self, img):
        try:
            result = self.detect_model(img).render()
            assert len(result) > 0
            result = result[0]
        except:
            result = img

        return result
    
    def get_frame(self):
        success, image = self.video.read()

        image = self.detect_target(image)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def StreamVideo(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
