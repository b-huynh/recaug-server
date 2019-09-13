import json
import struct

bufread = lambda buf, start, length: buf[start:start+length]

def pack(json_obj):
    json_bytes = json.dumps(json_obj).encode("utf-8")
    size = len(bytearray(json_bytes))
    size_bytes = struct.pack('<i', size)
    return size_bytes + json_bytes

def unpack(json_bytes):
    size = int.from_bytes(bufread(json_bytes, 0, 4), byteorder='little')
    json_string = bufread(json_bytes, 4, size).decode("utf-8")
    
    try:
        json_obj = json.loads(json_string)
        return json_obj
    except json.decoder.JSONDecodeError:
        print("SIZE: ", size)
        print(json_string)
        log_f = open("server_unpack.log", "w")
        log_f.write(json_string)
        log_f.write("\nSIZE: " + str(size))
        log_f.close()