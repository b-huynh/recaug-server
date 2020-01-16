import argparse
import functools
import io
import json
import os
import queue
import socket
import struct
import sys
import time
import urllib

import cv2
import numpy as np
from PIL import Image

# from recaug.models import ObjectDetector
# from recaug.server import AppLogServer, CameraFrameMessage, FrameServer, StaticServer, FPSTracker, MessageServer
# from recaug.server.utils import create_valid_message

import recaug.server as server
from recaug.models.threaded_object_detector_v2 import ThreadedObjectDetector
from recaug.predictions.predictions import PredictionsUtil
import recaug.predictions.filters as filters

CONFIG_URL = 'http://192.168.100.108:8080/config.json'
WINDOW_NAME = 'Hololens Webcam Frames (Debug)'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--debug', help='Enable debug view', action="store_true")


def record_message(userdata_path, message):
    session_path = os.path.join(userdata_path, message.header['sessionUUID'])
    if not os.path.isdir(session_path):
        os.mkdir(session_path)

    file_name = '{0:06d}.jpg'.format(message.header['frameID'])
    file_path = os.path.join(session_path, file_name)
    with open(file_path, 'wb') as f:
        f.write(message.payload)

def update_point_colors(out_message, predicted_points):
    for label_dict in predicted_points["labels"]:
        xcen = label_dict["xcen"]
        ycen = label_dict["ycen"]
        cen_r, cen_g, cen_b = out_message.frame[ycen][xcen]
        label_dict["cen_r"] = float(cen_r) / 255.0
        label_dict["cen_g"] = float(cen_g) / 255.0
        label_dict["cen_b"] = float(cen_b) / 255.0

def main():
    args = parser.parse_args()

    # Get config
    # result = urllib.request.urlopen(CONFIG_URL)
    result = open('config.json', 'rb')
    data = result.read().decode('utf-8')
    config = json.loads(data)

    # Utility Services
    app_log_port = int(config['System']['DebugLogPort'])
    app_log_server = server.AppLogServer('0.0.0.0', app_log_port)
    app_log_server.start()

    static_server = server.StaticServer('0.0.0.0', 8080)
    static_server.start()

    # The main message service
    message_port = int(config['System']['ObjectTrackingPort'])
    message_server = server.MessageServer('0.0.0.0', message_port)
    message_server.start()

    # Handle frame messages
    # confidence_threshold = config['System']['ConfidenceThreshold']
    model = ThreadedObjectDetector(
        'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')
    formatter = PredictionsUtil("TensorFlow2COCO",
        [
            filters.ConfidenceThreshold(0.5),
            filters.HighestConfidence()
        ]
    )

    debug_q = queue.Queue()
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def result_ready_handler(client_info, image, output, client_address):
        predictions = formatter.format(image, output)
        if client_address:
            # Send back to clients
            prediction_message = server.PredictionMessage(client_info)
            for p in predictions.predictions:
                prediction_message.add_prediction(p.class_name,
                    (p.xmin, p.xmax), (p.ymin, p.ymax), (0.0, 0.0, 0.0))

            packet = server.pack(prediction_message)
            # print(len(packet))
            send_sock.sendto(packet, client_address)
        if args.debug:
            debug_frame = image.copy()
            predictions.visualize(debug_frame)
            debug_q.put((debug_frame, predictions))

    def frame_message_handler(client_address, message):
        del message.json["frameBase64"]
        handler_with_info = functools.partial(result_ready_handler, message.json)
        model.enqueue(message.frame, client_address, handler_with_info)

    server.frame_message_event += frame_message_handler

    # rand_rgb = np.random.randint(255, size=(480,640,3),dtype=np.uint8)
    # cv2.imshow(WINDOW_NAME, rand_rgb)
    while True:
        if not debug_q.empty():
            frame, prediction = debug_q.get()
            prediction.visualize(frame)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass