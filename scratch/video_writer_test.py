import numpy as np
import cv2
import os
import sys
import time
import uuid

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    frame_idx = -1
    # fps = cam.get(cv2.CAP_PROP_FPS)
    # width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # video = cv2.VideoWriter()
    # # video.open("test.avi", -1, fps, (width, height), True)
    # video.open("test.avi", -1, 30, (640, 480), True)

    # logfile = open('test.vid', 'wb')

    session = str(uuid.uuid1())
    sess_path = os.path.join('data', session)
    os.mkdir(sess_path)

    try:
        while True:
            ret, frame = cam.read() # OpenCV natively uses numpy array
            frame_idx += 1
            ts = time.time() * 1000
            # logfile.write((ts, frame))
            # video.write(frame)
            fpath = os.path.join(sess_path, '{0:06d}.jpg'.format(frame_idx))
            cv2.imwrite(fpath, frame)
            cv2.imshow("test", frame)
            
            if cv2.waitKey(1) == ord('q'):
                # video.release()
                break
        
        # cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass