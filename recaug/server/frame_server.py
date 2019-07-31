import json
import os
import socket
import socketserver
import struct
import urllib

from .messages import CameraFrameMessage

class FrameServer:
    def __init__(self, config, netstats = None):
        # Initialize receive socket
        self.server_addr = ""
        self.server_port = int(config['System']['ObjectTrackingPort'])
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((self.server_addr, self.server_port))

        # Initialize send socket
        self.client_addr = config['System']['ClientIP']
        self.client_port = int(config['System']['ObjectTrackingPort'])
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # For Logging/Simulation
        self.userdata_path = os.path.join(os.getcwd(), 'userdata')
        self.netstats = netstats

    def recv_message(self):
        data, _ = self.recv_sock.recvfrom(CameraFrameMessage.MAX_PACKET_SIZE)

        message = CameraFrameMessage.from_bytes(data)
        message.update_timestamp('frameReceiveTimestamp')
        if self.netstats:
            self.netstats.update(message.payload_size)

        return message

    def send_message(self, message):
        message.update_timestamp('resultsSendTimestamp')
        message_bytes = message.to_bytes()
        self.send_sock.sendto(message_bytes, (self.client_addr, self.client_port))

