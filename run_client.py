import json
import socket
import sys
import urllib

import cv2

from server.messages import CameraFrameMessage

valid_header = {
    "type": "frame",
    "sessionUUID": "python_client",
    "payloadSize": 0
}

def create_valid_message(frame_bytes):
    message = CameraFrameMessage()
    valid_header['payloadSize'] = len(bytearray(frame_bytes))
    message.header = valid_header
    message.payload = frame_bytes
    return message

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)

    result = open('config.json', 'rb')
    data = result.read().decode('utf-8')
    config = json.loads(data)

    # Initialize send socket
    client_addr = '192.168.100.108'
    client_port = int(config['System']['ObjectTrackingPort'])
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
            
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        pass