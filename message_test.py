import base64
import json
import socket
import sys
import urllib

import cv2
import numpy as np

from recaug.server import CameraFrameMessage
from recaug.server.utils import create_valid_message

import recaug.server as server

def get_message_skeleton():
    skeleton = {
        "type": "frame",
        "sessionUUID": "d931eaf1-3ead-41fa-8ee1-fd4f2a045d1a",
        "metadataKeys": [],
        "metadataVals": [],
        "frameID": 12,
        "cameraMatrix": [
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612,
            0.10000000149011612
        ],
        "projectionMatrix": [
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421,
            0.8999999761581421
        ],
        "frameBase64": ""
    }
    return skeleton

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)

    # result = open('config.json', 'rb')
    # data = result.read().decode('utf-8')
    # config = json.loads(data)

    # Initialize send socket
    # client_addr = '192.168.100.108'
    # client_port = int(config['System']['ObjectTrackingPort'])
    # log_port = int(config['System']['DebugLogPort'])
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    log_f = open("log_message_test.txt", "w")

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
            # send_sock.sendto(message_bytes, (client_addr, client_port))
            # print("Num bytes original: ", len(message_bytes))

            encoded_bytes = base64.b64encode(frame_bytes)
            encoded_str = encoded_bytes.decode('utf-8')



            # encoded = encoded_bytes.decode()

            # Create pure JSON version.
            # header = {
            #     "type": "frame",
            #     "sessionUUID": "FakeMessage",
            #     "payloadSize": len(encoded),
            #     "payload": encoded
            # }

            skeleton = get_message_skeleton()
            skeleton["frameBase64"] = encoded_str
           
            frame_message = server.FrameMessage(skeleton)
            log_f.write(json.dumps(frame_message.json) + "\n")

            message_bytes = server.pack(frame_message)

            # jsb = json.dumps(header).encode('utf-8')
            # print(type(jsb))
            # print("Num bytes new: ", len(jsb))

            # ret = json.loads(jsb.decode('utf-8'))
            # img_bs = ret['payload'].encode()
            # bs_dec = base64.b64decode(img_bs)
            # orig_jpg = np.frombuffer(bs_dec, dtype=np.uint8)

            # print(type(orig_jpg))

            # orig_img = cv2.imdecode(orig_jpg, 1)

            # cv2.imshow("Test", orig_img)

            cv2.imshow("Test", frame)
            send_sock.sendto(message_bytes, ('192.168.100.233', 12000))

            if cv2.waitKey(30) == ord('q'):
                break

    except KeyboardInterrupt:
        pass