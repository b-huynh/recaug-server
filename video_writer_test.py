import numpy as np
import cv2
import sys

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_FPS, 30)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 504)

    try:
        while True:
            ret, frame = cam.read() # OpenCV natively uses numpy array
            
            if cv2.waitKey(1) == ord('q'):
                model.close()
                break
        
        # cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass