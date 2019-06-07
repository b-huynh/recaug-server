import numpy as np
import cv2
import sys

from trained_models.od_thread import ObjectDetector
from networking.messages import CameraFrameMessage

if __name__ == "__main__":
    model = ObjectDetector(threshold=0.85, single_instance=True)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 504)

    try:
        while True:
            ret, frame = cam.read() # OpenCV natively uses numpy array

            message = CameraFrameMessage()
            message.frame = frame

            model.enqueue(message)

            if model.result_ready:
                out_message, predictions = model.latest_result
                out_img = out_message.frame
                predicted_points = predictions.get_predicted_points()
                predictions.visualize(out_img)  # Draw bounding boxes

                cv2.imshow('Webcam (Debug)', out_img)

            # frame = cv2.flip(frame, 1)
            
            if cv2.waitKey(1) == ord('q'):
                model.close()
                break
        
        # cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass