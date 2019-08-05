import io
import json
import os
import socket
import struct
import sys
import time
import urllib

import cv2
import numpy as np
from PIL import Image

from recaug.models import ObjectDetector
from recaug.server import AppLogServer, CameraFrameMessage, FrameServer, StaticServer, FPSTracker
from recaug.server.utils import create_valid_message

CONFIG_URL = 'http://192.168.100.108:8080/config.json'
WINDOW_NAME = 'Hololens Webcam Frames (Debug)'

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
    # Get config
    # result = urllib.request.urlopen(CONFIG_URL)
    result = open('config.json', 'rb')
    data = result.read().decode('utf-8')
    config = json.loads(data)

    app_log_port = int(config['System']['DebugLogPort'])
    app_log_server = AppLogServer('0.0.0.0', app_log_port)
    app_log_server.start()

    static_server = StaticServer('0.0.0.0', 8080)
    static_server.start()

    confidence_threshold = config['System']['ConfidenceThreshold']
    model = ObjectDetector(
        'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        threshold=confidence_threshold, single_instance=True)
    frame_server = FrameServer(config)
    
    fps = FPSTracker()
    fps.start()

    # TensorFlow does lazy initialization, need to start computation early
    rand_rgb = np.random.randint(255, size=(480,640,3),dtype=np.uint8)
    rand_img = Image.fromarray(rand_rgb)
    rand_jpg = io.BytesIO()
    rand_img.save(rand_jpg, format='JPEG', quality=15)
    fake_message = create_valid_message(rand_jpg.getvalue())

    # TODO: Fix this silliness.
    msg_bytes = fake_message.to_bytes()
    msg = CameraFrameMessage.from_bytes(msg_bytes)

    model.enqueue(msg)

    while True:
        if model.result_ready:
            out_message, _ = model.latest_result
            cv2.imshow(WINDOW_NAME, out_message.frame)
            break

    # Begin processing client frames
    while True:  
        message = frame_server.recv_message()

        model.enqueue(message)

        if model.result_ready:
            out_message, predictions = model.latest_result
            predicted_points = predictions.get_predicted_points()

            # Add RGB to points
            update_point_colors(out_message, predicted_points)

            # Update timestamp
            out_message.update_timestamp('frameProcessedTimestamp')

            # Convert message type
            out_message.to_result(predicted_points)

            # Send the predictions back to client
            frame_server.send_message(out_message)

            # Debug Visualizations
            predictions.visualize(out_message.frame)
            
            # FPS
            fps.update_count()
            fps.visualize(out_message.frame)

            cv2.imshow(WINDOW_NAME, out_message.frame)    

        if cv2.waitKey(1) == ord('q'):
            model.close()
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass