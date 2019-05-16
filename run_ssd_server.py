import socket
import sys
import io
import time
import struct
import json

from PIL import Image
import cv2
import numpy as np

# sys.path.append('ssd_keras')
# from trained_models.coco_300 import Coco300
from trained_models.od_thread import ObjectDetector

# from predict_utils.predict_thread import PredictThread

MAX_PACKET_SIZE = 65536

# FOR RECEIVING
HOST, PORT = "", 12000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.settimeout(0.033) # 1/30th of a second for 30 FPS
sock.bind((HOST, PORT))


# FOR SENDING
SEND_IP, SEND_PORT = '192.168.2.238', 12000
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

def recv_single_packet_jpg(sock, netstats = None):
    data, _ = sock.recvfrom(MAX_PACKET_SIZE)
    
    # Index for current reading location in buffer
    ind = 0 
    # 4x4 matrix of 4 byte floats
    matrix_size = 64
    # Lambda for array slicing
    bufread = lambda buf, start, length: buf[start:start+length]

    # Read transformation matrices
    camera_world_matrix = bufread(data, ind, matrix_size)
    ind += matrix_size
    projection_matrix = bufread(data, ind, matrix_size)
    ind += matrix_size

    # Read jpg
    jpg_size = int.from_bytes(bufread(data, ind, 4), byteorder='little')
    ind += 4
    jpg_data = bufread(data, ind, jpg_size)
    jpg_stream = io.BytesIO(jpg_data)
    frame = np.array(Image.open(jpg_stream))

    # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if netstats:
        netstats.update(jpg_size)

    # Flip the image
    # frame = cv2.flip(frame, 0)
    return camera_world_matrix, projection_matrix, frame

def send_frame_predictions(camera_world_matrix, projection_matrix, predicted_points):
    pred_bytes = json.dumps(predicted_points).encode()
    pred_size = len(bytearray(pred_bytes))
    pred_size_bytes = struct.pack('<i', pred_size)
    message = camera_world_matrix + projection_matrix + pred_size_bytes + pred_bytes

    send_sock.sendto(message, (SEND_IP, SEND_PORT))    

def single_packet_loop(sock):
    model = ObjectDetector(threshold=0.85, single_instance=True)

    start_time = time.time()
    x = 1 # displays the frame rate every 1 second
    counter = 0
    fps = 0

    window_name = 'Hololens Webcam Frames (Debug)'
    # Show empty frame to start OpenCV render loop
    # black_frame = np.zeros((504, 896, 3)).astype(int)
    
    # cv2.imshow(window_name, black_frame)

    netstats = NetStatsTracker(check_rate=450)

    while True:  
        camera_world_matrix, projection_matrix, frame = recv_single_packet_jpg(
            sock, netstats=netstats)

        model.enqueue(frame)

        if model.result_ready:
            out_img, predictions = model.latest_result
            predicted_points = predictions.get_predicted_points()

            # Send the predictions back to client
            send_frame_predictions(camera_world_matrix, projection_matrix,
                predicted_points)

            # Debug Visualizations
            predictions.visualize(out_img)
            
            # FPS
            font = cv2.FONT_HERSHEY_SIMPLEX
            display_str = 'FPS: {:.1f}'.format(fps)
            cv2.putText(frame, display_str, (10,500), font, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, out_img)    
            counter += 1
            if (time.time() - start_time) > x :
                fps = counter / (time.time() - start_time)
                counter = 0
                start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            model.close()
            break

try:
    single_packet_loop(sock)
except KeyboardInterrupt:
    pass
