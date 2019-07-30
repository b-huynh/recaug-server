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

from server.frame_server import FrameServer
from server.static_server import StaticServer
from server.stats import FPSTracker, NetStatsTracker
from trained_models.od_thread import ObjectDetector

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
    static_server = StaticServer('0.0.0.0', 8080)
    static_server.start()

    # Get config
    # result = urllib.request.urlopen(CONFIG_URL)
    result = open('config.json', 'rb')
    data = result.read().decode('utf-8')
    config = json.loads(data)

    confidence_threshold = config['System']['ConfidenceThreshold']
    model = ObjectDetector(threshold=confidence_threshold, single_instance=True)
    frame_server = FrameServer(config)
    
    fps = FPSTracker()
    fps.start()
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