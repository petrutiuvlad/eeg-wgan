import struct
import sys
import numpy as np

PI_CORRELATION_WINDOW = 80
TRIAL = 200

def read_trial_floats(filename:str):
    result = []
    start_position = (2 * PI_CORRELATION_WINDOW + 1) * sys.getsizeof(float())
    start = start_position * (TRIAL - 1)
    end = start + start_position - 1
    pos = start
    with open(filename, "rb") as f:
        f.seek(start, 0)
        i = 0
        while pos < end:
            data = f.read(4)
            if data == b'':
                break
            result.append(struct.unpack('f', data)[0])
            i += 1
            pos += sys.getsizeof(float())

    return result

def read_floats(filename:str):
    result = []
    position = 0
    with open(filename, "rb") as f:
        k = struct.unpack('f', f.read(4))[0]
        result.append(k)
        position += 1
        while k != b'':
            data = f.read(4)
            if data == b'':
                break
            k = struct.unpack('f', data)[0]
            result.append(k)
            position += 1
    return result

def read_signal(filename:str):
    signal = {}
    result = read_floats(filename)
    for i in range(0, len(result)):
        signal.update({i: result[i]})
    return signal

def read_event(filename:str):
    result = []
    position = 0
    with open(filename, "rb") as f:
        k = struct.unpack('i', f.read(4))[0]
        result.append(k)
        position += 1
        while k != b'':
            data = f.read(4)
            if data == b'':
                break
            k = struct.unpack('i', data)[0]
            result.append(k)
            position += 1
    return result

def read_timestamp(filename:str):
    result = []
    position = 0
    with open(filename, "rb") as f:
        k = struct.unpack('i', f.read(4))[0]
        result.append(k)
        position += 1
        while k != b'':
            data = f.read(4)
            if data == b'':
                break
            k = struct.unpack('i', data)[0]
            result.append(k)
            position += 1
    return result