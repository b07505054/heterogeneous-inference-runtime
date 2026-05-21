import cv2


class VideoFrameSource:
    def __init__(self, source: str | int):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        self.cap.release()