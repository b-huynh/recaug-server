from .messages import CameraFrameMessage

valid_header = {
    "type": "frame",
    "sessionUUID": "FakeMessage",
    "payloadSize": 0
}

def create_valid_message(frame_bytes):
    message = CameraFrameMessage()
    valid_header['payloadSize'] = len(bytearray(frame_bytes))
    message.header = valid_header
    message.payload = frame_bytes
    return message