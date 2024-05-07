#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import struct
import numpy as np
import plotly.graph_objects as go
import ROOT
from tqdm import tqdm


def get_args():
    parser = ArgumentParser(description='Digital Board Reader')
    parser.add_argument('filename', type=Path, help='Input binary file')
    parser.add_argument('-c', '--config', type=Path, default='share/config.yaml')
    parser.add_argument('-m', '--modes', default='wave', nargs='+', choices=['wave', 'root', 'debug'], help='Modes')
    return parser.parse_args()


def find_sublist(lst, sublst):
    # Get the sliding window view
    sliding_view = np.lib.stride_tricks.sliding_window_view(lst, len(sublst))
    # Check where the sublist matches
    matching_indices = np.where((sliding_view == sublst).all(axis=1))[0]
    if matching_indices.size > 0:
        # Return the first matching index
        return matching_indices[0]
    else:
        # Return -1 if no match is found
        return -1

def check_header(header_unpacked):
    if header_unpacked[3] != 0x2104:
        raise ValueError('Invalid header')
    if header_unpacked[4] > 8:
        raise ValueError('Invalid header: channel too large')


def check_data(data_unpacked):
    if np.max(data_unpacked) > 16384:
        raise ValueError(f'Invalid data: too large at {hex(np.max(data_unpacked))}')


def read_packets(file_path, start_offset, modes):
    packets = {
        "length_buff": [],
        "data_length": [],
        "offset_buff": [],
        "threshold_buff": [],
        "id": [],
        "time_tick": [],
        "data": [],
        "mean": [],
        "std": [],
        "max": [],
        "min": []
    }

    with open(file_path, 'rb') as file:
        file_size = Path(file_path).stat().st_size
        header_offset = start_offset
        prev_header_invalid = False
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Reading') as pbar:
            pbar.update(header_offset)
            while True:
                # ===================================================
                # Read header
                # ---------------------------------------------------
                header_length = 16
                file.seek(header_offset)
                header = file.read(header_length)
                if not header or len(header) != header_length:
                    break
                header_unpacked = struct.unpack('>8H', header)
                header_str = ""
                for number in header_unpacked:
                    header_str += f"{number:04x} "
 
                try:
                    check_header(header_unpacked)
                except ValueError as e:
                    if not prev_header_invalid:
                        print('\033[91m' + str(e), ":", hex(header_offset), ":", header_str)
                        prev_header_invalid = True
                    header_offset += 8
                    pbar.update(8)
                    continue
                if prev_header_invalid:
                    print('\033[92mRecovered at   :', hex(header_offset), ":", header_str)
                    prev_header_invalid = False
                # Output header info
                length_offset = header_unpacked[3]
                offset_buff = length_offset & 0xFF
                length_buff = (length_offset >> 8) & 0xFF
               
                # ===================================================
                # Read data
                # ---------------------------------------------------
                data_length = length_buff * 8
                data_offset = header_offset + header_length
                file.seek(data_offset)
                data = file.read(data_length)
                if not data or len(data) != data_length:
                    break
                num_data = len(data) // 2
                data_unpacked = np.asarray(struct.unpack(f'>{num_data}H', data))
                header_pos = find_sublist(data_unpacked, np.array([0x2104, 0x0000]))
                if header_pos >= 0:
                    if header_pos > 3:
                        # Reset data_length to the next header and trim the data_unpacked
                        data_n = header_pos - 3
                        data_length = data_n * 2
                        data_unpacked = data_unpacked[:data_n]
                    else:
                        print('\033[91m' + hex(data_offset), ": no data")
                        continue

                try:
                    check_data(data_unpacked)
                except ValueError as e:
                    print('\033[91m' + hex(data_offset), ":", e)
                    pbar.update(data_offset + data_length - header_offset)
                    header_offset = data_offset + data_length
                    prev_header_invalid = True
                    continue
                # ===================================================
                # Record data
                # ---------------------------------------------------
                packets["offset_buff"].append(offset_buff)
                packets["data_length"].append(data_length)
                packets["length_buff"].append(length_buff)
                packets["threshold_buff"].append(header_unpacked[2])
                packets["id"].append(header_unpacked[1])
                packets["time_tick"].append(header_unpacked[7])
                packets["mean"].append(np.mean(data_unpacked))
                packets["std"].append(np.std(data_unpacked))
                packets["max"].append(np.max(data_unpacked))
                packets["min"].append(np.min(data_unpacked))
                packets["data"].append(data_unpacked)
                if 'wave' in modes or 'debug' in modes:
                    if len(packets["data"]) >= 100:
                         break

                # ===================================================
                # Move to the next packet
                # ---------------------------------------------------
                pbar.update(data_offset + data_length - header_offset)
                header_offset = data_offset + data_length
    return packets

if __name__ == "__main__":
    # ===================================================
    # Initialize varaibles
    # ---------------------------------------------------
    args = get_args()
    file_path = args.filename
    modes = args.modes
    start_offset = 0
    sampling_interval = 1 # unit ns
    sampling_rate = 1 / sampling_interval # unit GHz
    # ===================================================
    # Processing binary file
    # ---------------------------------------------------
    packets = read_packets(file_path, start_offset, modes)

    # ===================================================
    # Output root file
    # ---------------------------------------------------
    if 'root' in modes:
        print("Saving root file")
        root_file_path = file_path.with_suffix('.root')
        # Create RDataFrame and Write to file
        df = ROOT.RDF.FromNumpy({key: np.asarray(value) for key, value in packets.items() if key != 'data'})
        df.Snapshot("tree", str(root_file_path))

    # ===================================================
    # Plotting and output
    # ---------------------------------------------------
    # Plotting each packet data as a separate trace
    if 'wave' in modes:
        fig = go.Figure()
        for i, packet in enumerate(packets["data"]):
            fig.add_trace(
                go.Scatter(
                    y=packet,
                    mode='lines+markers',
                    name=f'id: {packets["id"][i]}, time tick: {packets["time_tick"][i]}'
                )
            )
        
        fig.add_annotation (
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            text=f'Sampling rate: {sampling_rate} GHz<br>Offset: {packets["offset_buff"][0] * sampling_interval * 4} ns<br>Threshold: {packets["threshold_buff"][0]}',
            showarrow=False,
            font_size=20,
            align="left",
        )
        
        fig.update_layout(title='Packet Data Plot', xaxis_title='Time (ns)', yaxis_title='Value')
        fig.show()

