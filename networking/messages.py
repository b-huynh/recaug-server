import io
import json
import struct
import time

import cv2
import numpy as np
from PIL import Image


class CameraFrameMessage:
    def __init__(self):
        self.type = None
        self.header = None
        self.payload = None
        self.payload_size = None
        self.frame = None
        self.predicted_points = None

    @classmethod
    def from_bytes(cls, data):
        retval = cls()

        # Helpers
        idx = 0
        bufread = lambda buf, start, length: buf[start:start+length]

        # Read header for message size
        json_sz = int.from_bytes(bufread(data, idx, 4), byteorder='little')
        idx += 4

        # Read message as JSON
        json_raw = bufread(data, idx, json_sz).decode("utf-8")
        idx += json_sz
        retval.header = json.loads(json_raw)
        retval.type = retval.header["type"]

        # Read payload
        payload_sz = retval.header["payloadSize"]
        retval.payload = bufread(data, idx, payload_sz)
        retval.payload_size = payload_sz
        
        # Convert to frame
        jpg_stream = io.BytesIO(retval.payload)
        np_frame = np.array(Image.open(jpg_stream))
        retval.frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB) # BGR to RGB

        return retval

    def update_header(self, key, ts):
        self.header[key] = ts

    def update_timestamp(self, key):
        self.update_header(key, time.time() * 1000)
        
    # Switch message types (replaces payload)
    def to_result(self, predicted_points):
        self.predicted_points = predicted_points
        self.payload = json.dumps(self.predicted_points).encode()
        self.payload_size = len(bytearray(self.payload))
        self.header['payloadSize'] = self.payload_size
        self.header['results'] = predicted_points
        self.type = "result"

    def to_bytes(self):
        # Convert header to bytes, get it's size in bytes
        header_bytes = json.dumps(self.header).encode()
        header_size = len(bytearray(header_bytes))
        header_size_bytes = struct.pack('<i', header_size)
        
        return header_size_bytes + header_bytes + self.payload