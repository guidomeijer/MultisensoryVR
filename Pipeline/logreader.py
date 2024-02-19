# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:34:17 2023

@author: Neuropixel
"""

import numpy as np
import base64
import struct
from cobs import cobs
from tqdm.notebook import tqdm
from collections import namedtuple


def invert_polarity(digital_channel):
    '''
    Inverts channel polarity.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Digital channel data with inverted polarity.
    '''
    digital_channel = np.logical_not(digital_channel).astype(int)
    return digital_channel


def compute_onsets(digital_channel):
    '''
    Compute transitions 0->1 in digital channel.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Indices of transitions from 0 to 1 in the digital channel.
    '''
    digital_channel = digital_channel.astype(int)
    onsets = np.where(np.diff(digital_channel) == 1)[0]+1
    return onsets


def compute_offsets(digital_channel):
    '''
    Compute transitions 1->0 in digital channel.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Indices of transitions from 1 to 0 in the digital channel.
    '''
    digital_channel = digital_channel.astype(int)
    onsets = np.where(np.diff(digital_channel) == -1)[0]
    return onsets


def compute_switch(digital_channel):
    '''
    Compute all transitions in the digital channel.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Indices of all transitions in the digital channel.
    '''
    digital_channel = digital_channel.astype(int)
    onsets = np.where(np.diff(digital_channel) != 0)[0]
    return onsets

def count_lines(fp):
    '''
    Counts the number of lines in a b64 file.

    Parameters:
        fp (str): Path to the file.

    Returns:
        int: Number of lines in the file.
    '''
    def _make_gen(reader):
        b = reader(2**16)
        while b:
            yield b
            b = reader(2**16)
    with open(fp, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count


def unpack_data_packet(dp, DataPacketStruct, DataPacket):
    '''
    Unpacks the data packet from b64 file stream.

    Parameters:
        dp (bytes): Data packet to unpack.
        DataPacketStruct (str): Structure of the data packet.
        DataPacket (namedtuple): Namedtuple representing the data packet structure.

    Returns:
        namedtuple: Unpacked data packet.
    '''
    s = struct.unpack(DataPacketStruct, dp)
    up = DataPacket(type=s[0], size=s[1], crc16=s[2], packetID=s[3], us_start=s[4], us_end=s[5],
                    analog=s[6:14], states=s[14:22], digitalIn=s[22], digitalOut=s[23], padding=None)
    return up


def create_bp_structure(bp):
    '''
    Decode the log file and create a VR channel data structure.

    Parameters:
        bp (str): Path to the log file to read.

    Returns:
        dict: A dictionary containing the decoded log data with the following keys:
            - 'analog': Analog data.
            - 'digitalIn': Digital input data (flipped left to right).
            - 'digitalOut': Digital output data (flipped left to right).
            - 'startTS': Start timestamps in microseconds.
            - 'transmitTS': Transmit timestamps in microseconds.
            - 'longVar': States data.
            - 'packetNums': Packet IDs.

    Note:
        This function decodes will soon be updated with a better name and a more flexible logic.
    '''

    print('Decoding log file')
    # Format package
    DataPacketDesc = {'type': 'B',
                      'size': 'B',
                      'crc16': 'H',
                      'packetID': 'I',
                      'us_start': 'I',
                      'us_end': 'I',
                      'analog': '8H',
                      'states': '8l',
                      'digitalIn': 'H',
                      'digitalOut': 'B',
                      'padding': 'x'}

    DataPacket = namedtuple('DataPacket', DataPacketDesc.keys())
    DataPacketStruct = '<' + ''.join(DataPacketDesc.values())
    DataPacketSize = struct.calcsize(DataPacketStruct)

    # package with non-digital data
    dtype_no_digital = [
        ('type', np.uint8),
        ('size', np.uint8),
        ('crc16', np.uint16),
        ('packetID', np.uint32),
        ('us_start', np.uint32),
        ('us_end', np.uint32),
        ('analog', np.uint16, (8, )),
        ('states', np.uint32, (8, ))]

    # DigitalIn and DigitalOut
    dtype_w_digital = dtype_no_digital + \
        [('digital_in', np.uint16, (16, )), ('digital_out', np.uint8, (8, ))]

    # Creating array with all the data (differenciation digital/non digital)
    np_DataPacketType_noDigital = np.dtype(dtype_no_digital)
    np_DataPacketType_withDigital = np.dtype(dtype_w_digital)
    # Unpack the data as done on the teensy commander code
    num_lines = count_lines(bp)
    log_duration = num_lines/1000/60

    # Decode and create new dataset
    data = np.zeros(num_lines, dtype=np_DataPacketType_withDigital)
    non_digital_names = list(np_DataPacketType_noDigital.names)

    with open(bp, 'rb') as bf:
        for nline, line in enumerate(tqdm(bf, total=num_lines)):
            bl = cobs.decode(base64.b64decode(line[:-1])[:-1])
            dp = unpack_data_packet(bl, DataPacketStruct, DataPacket)

            data[non_digital_names][nline] = np.frombuffer(
                bl[:-4], dtype=np_DataPacketType_noDigital)
            digital_arr = np.frombuffer(bl[-4:], dtype=np.uint8)
            data[nline]['digital_in'] = np.hstack(
                [np.unpackbits(digital_arr[1]), np.unpackbits(digital_arr[0])])
            data[nline]['digital_out'] = np.unpackbits(
                np.array(digital_arr[2], dtype=np.uint8))
        # Check for packetID jumps
    jumps = np.unique(np.diff(data['packetID']))
    decoded = {"analog": data['analog'], "digitalIn": data['digital_in'][:, ::-1], "digitalOut": data['digital_out'][:, ::-1],
               "startTS": data['us_start'], "transmitTS": data['us_end'], "longVar": data['states'], "packetNums": data['packetID']}

    return decoded