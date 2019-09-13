import base64
import io
import json
import struct
import time

import cv2
import numpy as np
from PIL import Image

bufread = lambda buf, start, length: buf[start:start+length]

def pack(msg):
    json_bytes = json.dumps(msg.json).encode("utf-8")
    size = int(len(bytearray(json_bytes)))
    size_bytes = struct.pack('<i', size)
    return size_bytes + json_bytes

def unpack(json_bytes):
    size = int.from_bytes(bufread(json_bytes, 0, 4), byteorder='little')
    json_string = bufread(json_bytes, 4, size).decode("utf-8")
    json_obj = json.loads(json_string)

    message_type = json_obj["type"]
    if message_type == "frame":
        return FrameMessage(json_obj)
    else:
        return Message(json_obj)

sample_json = 'C:\\Users\\peter\\Projects\\recaug-uwp-client\\Assets\\test.json'

class Message:
    """
    Do not instantiate yourself. Use unpack to create Message from byte array.
    """
    def __init__(self, json_obj):
        self.json = json_obj
    
    @property
    def type(self):
        return self.json["type"]

    @property
    def sessionUUID(self):
        return self.json["sessionUUID"]

    def get_metadata(self, key):
        idx = self.json["metadataKeys"].index(key)
        return self.json["metadataVals"][idx]
    
    def set_metadata(self, key, value):
        if not isinstance(key, str):
            raise TypeError("`key` must be a string")
        if not isinstance(value, str):
            raise TypeError("`value` must be a string")

        try:
            idx = self.json["metadataKeys"].index(key)
            self.json["metadataVals"][idx] = value
        except ValueError:
            self.json["metadataKeys"].append(key)
            self.json["metadataVals"].append(value)

    def __str__(self):
        return str(self.json)
    
    def __repr__(self):
        return repr(self.json)

class FrameMessage(Message):
    def __init__(self, json_obj):
        super().__init__(json_obj)
        base64_str = self.json["frameBase64"]
        jpg_bytes = io.BytesIO(base64.b64decode(base64_str))
        np_frame = np.array(Image.open(jpg_bytes))
        self._frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB) # BGR to RGB

    @property
    def frame(self):
        return self._frame

class PredictionMessage(Message):
    def __init__(self, json_obj):
        super().__init__(json_obj)

    @property
    def predictions(self):
        return self.json["predictions"]
    
    def add_prediction(self, name, x, y, rgb):
        xcen = int((x[1] - x[0]) / 2)
        ycen = int((y[1] - y[0]) / 2)
        p = {
            "className": name,
            "xmin": x[0],
            "xmax": x[1],
            "ymin": y[0],
            "ymax": y[1],
            "xcen": xcen,
            "ycen": ycen,
            "cen_r": rgb[0],
            "cen_g": rgb[1],
            "cen_b": rgb[2]
        }
        self.predictions.add(p)

    def remove_prediction(self, idx):
        del self.predictions[idx]

class CameraFrameMessage:
    """ 
    Class that encapsulates encoding, decoding, and updating message data about
    a single camera frame from client. A message is a structured byte array of 
    the form:

    [ header size (4 byte integer) ][ header (encoded JSON) ][ payload (bytes) ]
 
    The header MUST include the following fields:
        type: (string), either "frame" or "result"
        sessionUUID: (string)
        payloadSize: (int), corresponding the the size in bytes of the payload
    """
    
    MAX_PACKET_SIZE = 65536

    def __init__(self):
        self.header = None
        self.payload = None
        self.payload_size = None

        # These are convenience variables...
        self.type = None
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