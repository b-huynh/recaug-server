import numpy as np
import cv2
import sys

sys.path.append('ssd_keras')
from trained_models.coco_300 import Coco300

if __name__ == "__main__":
    model = Coco300()

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 20)

    try:
        while True:
            ret, img = cam.read() # OpenCV natively uses numpy array

            if img is None:
                print("Not receiving frames...")
                continue

            img = cv2.flip(img, 1)

            # Predict on image
            coco_predictions = model.predict(img)

            # Visualize on original image
            coco_predictions.visualize(img)

            cv2.imshow('Predictions', img)
            # cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass