import base64
import json
import socket
import sys

import cv2

import recaug.server as server

class SimpleFrameMessageFactory:    
    """Constructs barebones FrameMessage object, useful for local testing."""
    def __init__(self, quality=30):
        self._frame_id = 0
        self._quality = quality
        self._json_template = {
            "type": "frame",
            "sessionUUID": "recaug-python-client",
            "metadataKeys": [],
            "metadataVals": [],
            "frameID": -1,
            "cameraMatrix": [0.0] * 16,
            "projectionMatrix": [0.0] * 16,
            "frameBase64": ""
        }


    def get_message(self, frame):
        """Creates a simple FrameMessage object using a 3-channel numpy image.
            
            Args:
                frame: 3-channel numpy array 
            Returns:
                A FrameMessage object.
        """
        # Encode frame as JPG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        _, jpg = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = jpg.tobytes()

        # Base64 encode the frame and convert the bytes to a string
        encoded_bytes = base64.b64encode(frame_bytes)
        encoded_str = encoded_bytes.decode('utf-8')
        
        # Add encoded frame to JSON skeleton
        self._frame_id += 1
        self._json_template["frameID"] = self._frame_id
        self._json_template["frameBase64"] = encoded_str
        
        # Create FrameMessage object from JSON
        return server.FrameMessage(self._json_template)
    

if __name__ == "__main__":
    # Get camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)

    # Open UDP outgoing socket
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Set up message factory
    message_factory = SimpleFrameMessageFactory(quality=30)
    try:
        while True:
            # Read from webcam, OpenCV natively returns numpy arrays
            _, frame = cam.read() 

            # Exit if not receiving frames due to broken webcam etc...
            if frame is None:
                print("No frames received...")
                sys.exit()

            # Construct FrameMessage
            frame_message = message_factory.get_message(frame)

            # Pack FrameMessage into bytes
            message_bytes = server.pack(frame_message)

            # Display debug frames
            cv2.imshow("Client Outgoing Webcam Frames", frame)

            # Send Frames
            send_sock.sendto(message_bytes, ('192.168.100.233', 12000))

            if cv2.waitKey(30) == ord('q'):
                break

    except KeyboardInterrupt:
        pass