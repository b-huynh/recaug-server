import json
import socket
import sys
import urllib

import cv2

from recaug.server import CameraFrameMessage
from recaug.server.utils import create_valid_message

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 504)

    result = open('config.json', 'rb')
    data = result.read().decode('utf-8')
    config = json.loads(data)

    # Initialize send socket
    client_addr = '192.168.100.108'
    client_port = int(config['System']['ObjectTrackingPort'])
    log_port = int(config['System']['DebugLogPort'])
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def debug_log(s):
        send_sock.sendto(s.encode('utf-8'), (client_addr, log_port))

    debug_log("Starting Python Client")

    try:
        while True:
            _, frame = cam.read() # OpenCV natively uses numpy array

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            _, jpg = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = jpg.tobytes()

            if frame is None:
                print("No frames received...")
                sys.exit()
            
            message = create_valid_message(frame_bytes)
            message_bytes = message.to_bytes()
            send_sock.sendto(message_bytes, (client_addr, client_port))
            debug_log("Sent {} bytes".format(len(message_bytes)))
            
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        pass