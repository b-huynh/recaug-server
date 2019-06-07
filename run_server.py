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

from trained_models.od_thread import ObjectDetector
from networking.messages import CameraFrameMessage

MAX_PACKET_SIZE = 65536

# GET CONFIG
CONFIG_URL = 'http://192.168.2.134:8080/config.json'
result = urllib.request.urlopen(CONFIG_URL)
data = result.read().decode('utf-8')
CONFIG = json.loads(data)

# FOR RECEIVING
SERVER_ADDR = ""
SERVER_PORT = int(CONFIG['System']['ObjectTrackingPort'])
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind((SERVER_ADDR, SERVER_PORT))

# FOR SENDING
CLIENT_ADDR = CONFIG['System']['ClientIP']
CLIENT_PORT = int(CONFIG['System']['ObjectTrackingPort'])
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Object Detection Params
CONFIDENCE_THRESHOLD = CONFIG['System']['ConfidenceThreshold']

# For Logging/Simulation
USERDATA_PATH = os.path.join(os.getcwd(), 'userdata')

class NetStatsTracker(object):
    def __init__(self, check_rate=900):
        self._check_rate = check_rate
        self._check_idx = 0
        self._jpg_running_total = 0
        self._avg_jpg_size = 0

    def update(self, jpg_size):
        self._check_idx += 1
        self._jpg_running_total += jpg_size
        if self._check_idx % self._check_rate == 0:
            self._avg_jpg_size = self._jpg_running_total / self._check_rate
            self._check_idx = 0
            self._jpg_running_total = 0
            print("[NetStats] Avg JPG Size: {:.2f} bytes".format(self._avg_jpg_size))
 
    @property
    def avg_jpg_size(self):
        return self._avg_jpg_size

def recv_single_packet_jpg(recv_sock, netstats = None):
    data, _ = recv_sock.recvfrom(MAX_PACKET_SIZE)
    
    # # Index for current reading location in buffer
    # ind = 0 
    # # 4x4 matrix of 4 byte floats
    # matrix_size = 64
    # # Lambda for array slicing
    # bufread = lambda buf, start, length: buf[start:start+length]

    # # Read transformation matrices
    # camera_world_matrix = bufread(data, ind, matrix_size)
    # ind += matrix_size
    # projection_matrix = bufread(data, ind, matrix_size)
    # ind += matrix_size

    # # Read jpg
    # jpg_size = int.from_bytes(bufread(data, ind, 4), byteorder='little')
    # ind += 4
    # jpg_data = bufread(data, ind, jpg_size)
    # jpg_stream = io.BytesIO(jpg_data)
    # frame = np.array(Image.open(jpg_stream))

    # # Convert from BGR to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if netstats:
    #     netstats.update(jpg_size)

    # Flip the image
    # frame = cv2.flip(frame, 0)

    message = CameraFrameMessage.from_bytes(data)
    if netstats:
        netstats.update(message.payload_size)

    camera2world = message.header["camera2World"]
    projection = message.header["projection"]

    return camera2world, projection, message.frame

def send_frame_predictions(camera_world_matrix, projection_matrix, predicted_points):
    pred_bytes = json.dumps(predicted_points).encode()
    pred_size = len(bytearray(pred_bytes))
    pred_size_bytes = struct.pack('<i', pred_size)
    message = camera_world_matrix + projection_matrix + pred_size_bytes + pred_bytes

    send_sock.sendto(message, (CLIENT_ADDR, CLIENT_PORT))    

def recv_message(recv_sock, netstats = None):
    data, _ = recv_sock.recvfrom(MAX_PACKET_SIZE)

    message = CameraFrameMessage.from_bytes(data)
    message.update_timestamp('frameReceiveTimestamp')
    if netstats:
        netstats.update(message.payload_size)
    
    session_path = os.path.join(USERDATA_PATH, message.header['sessionUUID'])
    if not os.path.isdir(session_path):
        os.mkdir(session_path)

    file_name = '{0:06d}.jpg'.format(message.header['frameID'])
    file_path = os.path.join(session_path, file_name)
    with open(file_path, 'wb') as f:
        f.write(message.payload)

    return message

def send_message(message):
    message.update_timestamp('resultsSendTimestamp')
    message_bytes = message.to_bytes()
    send_sock.sendto(message_bytes, (CLIENT_ADDR, CLIENT_PORT))

def single_packet_loop(sock):
    model = ObjectDetector(threshold=CONFIDENCE_THRESHOLD, single_instance=True)

    start_time = time.time()
    x = 1 # displays the frame rate every 1 second
    counter = 0
    fps = 0

    window_name = 'Hololens Webcam Frames (Debug)'
    # Show empty frame to start OpenCV render loop
    # black_frame = np.zeros((504, 896, 3)).astype(int)
    
    # cv2.imshow(window_name, black_frame)

    netstats = NetStatsTracker(check_rate=450)

    # Write bytes to file

    while True:  
        # camera_world_matrix, projection_matrix, frame = recv_single_packet_jpg(
        #     sock, netstats=netstats)

        message = recv_message(sock, netstats=netstats)

        model.enqueue(message)

        if model.result_ready:
            out_message, predictions = model.latest_result
            predicted_points = predictions.get_predicted_points()

            # Add RGB to points
            for label_dict in predicted_points["labels"]:
                xcen = label_dict["xcen"]
                ycen = label_dict["ycen"]
                cen_r, cen_g, cen_b = out_message.frame[ycen][xcen]
                label_dict["cen_r"] = float(cen_r) / 255.0
                label_dict["cen_g"] = float(cen_g) / 255.0
                label_dict["cen_b"] = float(cen_b) / 255.0

            out_message.update_timestamp('frameProcessedTimestamp')
            out_message.to_result(predicted_points)
            # Send the predictions back to client
            # send_frame_predictions(camera_world_matrix, projection_matrix,
            #     predicted_points)
            
            send_message(out_message)

            # Debug Visualizations
            predictions.visualize(out_message.frame)
            
            # FPS
            font = cv2.FONT_HERSHEY_SIMPLEX
            display_str = 'FPS: {:.1f}'.format(fps)
            cv2.putText(out_message.frame, display_str, (10,500), font, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, out_message.frame)    
            counter += 1
            if (time.time() - start_time) > x :
                fps = counter / (time.time() - start_time)
                counter = 0
                start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            model.close()
            break

if __name__ == '__main__':
    try:
        single_packet_loop(recv_sock)
    except KeyboardInterrupt:
        pass