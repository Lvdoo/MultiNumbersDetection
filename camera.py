import cv2 as cv


def open_camera():
    video = cv.VideoCapture(1)
    
    video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not video.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    return video


def read_frame(video):
    ret, frame = video.read()

    if not ret:
        return False, None

    return True, frame


def release_video(video):
    video.release()
    cv.destroyAllWindows()