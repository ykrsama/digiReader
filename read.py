#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import struct
import numpy as np
import plotly.graph_objects as go
import ROOT
from tools import read_bin

MAX_DATA_VALUE = 16384
HEADER_LENGTH = 16
HEADER_FIND_STEP = 8


def get_args():
    parser = ArgumentParser(description='Digital Board Reader')
    parser.add_argument('filename', type=Path, help='Input binary file')
    parser.add_argument('-m', '--modes',
                        default=['wave'],
                        nargs='+',
                        choices=['wave', 'root', 'denoise', 'gmm', 'landau'],
                        help='Modes')
    parser.add_argument('-i', '--id', nargs='+', type=int, default=[None], help='id begin, id end')
    parser.add_argument('-b', '--buff', nargs='?', type=lambda x: int(x, 0), default=None, help='buffer_info, i.e. 0x2104')
    return parser.parse_args()


def read_packets(file_path, begin_id, end_id, buff_info, cfgs):
    packets = {key: [] for key in ["length_buff", "data_length", "offset_buff", "threshold_buff", "channel_n",
                                   "id", "time_tick",
                                   "data", "mean", "std", "max", "min",
                                   "baseline_median", "net_signal_median", "net_signal_denoised_median",
                                   "baseline_landau", "net_signal_landau", "net_signal_denoised_landau",
                                   "baseline_gmm", "net_signal_gmm", "net_signal_denoised_gmm"
                                   ]}
    prev_header_valid = True

    with open(file_path, 'rb') as file:
        if not begin_id:
            begin_id = 1
        if buff_info:
            print(f'Finding header with id {begin_id} and buff_info {hex(buff_info)}')
        else:
            print(f'Finding header with id {begin_id}')
        header_offset, header_info = read_bin.find_begin_header_info(file, begin_id, buff_info, HEADER_LENGTH, HEADER_FIND_STEP)
        print(f'Found first header: {header_info["str"]}')
        header_pattern = (header_info["length_offset"], header_info["channel_n"])
        with tqdm(total=file_path.stat().st_size, unit='B', unit_scale=True, desc='Reading') as pbar:
            pbar.update(header_offset)
            while True:
                # ===================================================
                # Process header and data, handle exception
                # ---------------------------------------------------
                # Process header
                try:
                    header_info, data_offset, data_length = read_bin.process_header(file, header_offset, HEADER_LENGTH, header_pattern)
                except IOError:
                    break
                except ValueError as e:
                    if prev_header_valid:
                        print('\033[91m' + str(e), hex(header_offset), ":", header_info["str"])
                        prev_header_valid = False
                    header_offset += HEADER_FIND_STEP
                    pbar.update(HEADER_FIND_STEP)
                    continue

                if not prev_header_valid:
                    print('\033[92mFind header at', hex(header_offset), ":", header_info["str"])
                    prev_header_valid = True

                if 'wave' in cfgs["modes"]:
                    if len(packets["id"]) >= 100:
                        break

                if end_id and header_info["id"] >= end_id:
                    break

                # Process Data
                try:
                    result, data_length, data_end_offset = read_bin.process_data(file, data_offset, data_length, header_pattern, cfgs)
                except IOError:
                    break
                except ValueError as e:
                    print('\033[91m' + hex(data_offset), ":", e)
                    pbar.update(data_offset + data_length - header_offset)
                    header_offset = data_offset + data_length
                    prev_header_valid = False
                    continue

                # ===================================================
                # Record header and data
                # ---------------------------------------------------
                packets["data_length"].append(data_length)
                for key, value in header_info.items():
                    if key == "length_offset" or key == "str":
                        continue
                    packets[key].append(value)

                for key, value in result.items():
                    packets[key].append(value)

                # ===================================================
                # Move to the next packet
                # ---------------------------------------------------
                pbar.update(data_end_offset - header_offset)
                header_offset = data_end_offset

    return packets

if __name__ == "__main__":
    # ===================================================
    # Initialize varaibles
    # ---------------------------------------------------
    cfgs = {
        "denoise_savgol_window_length": 37,
        "denoise_savgol_polyorder": 5,
        "gmm_n_components": 9,
    }
    start_offset = 0
    sampling_interval = 1 # unit ns
    sampling_rate = 1 / sampling_interval  # unit GHz
    # Read Arguments
    args = get_args()
    file_path = args.filename
    cfgs["modes"] = args.modes
    ids = args.id
    buff = args.buff
    begin_id = ids[0] if len(ids) > 0 else None
    end_id = ids[1] if len(ids) > 1 else None
    # ===================================================
    # Process binary file
    # ---------------------------------------------------
    packets = read_packets(file_path, begin_id, end_id, buff, cfgs)

    # ===================================================
    # Output root file
    # ---------------------------------------------------
    if 'root' in cfgs["modes"]:
        root_file_path = file_path.with_suffix('.root')

        if 'wave' in cfgs["modes"]:
            root_file_path = root_file_path.with_name(root_file_path.stem + '.wave.root')
        if begin_id:
            root_file_path = root_file_path.with_name(root_file_path.stem + f'.id{begin_id}_.root')
            if end_id:
                root_file_path = root_file_path.with_name(root_file_path.stem + f'{end_id}.root')

        print(f"Creating root file: {root_file_path}")
        # Create RDataFrame and Write to file
        df = ROOT.RDF.FromNumpy({key: np.asarray(value) for key, value in packets.items() if key != 'data' and value})
        df.Snapshot("tree", str(root_file_path))

    # ===================================================
    # Plotting
    # ---------------------------------------------------
    # Plotting each packet data as a separate trace
    if 'wave' in cfgs["modes"]:
        fig = go.Figure()

        for i, packet in enumerate(packets["data"]):
            fig.add_trace(
                go.Scatter(
                    y=packet,
                    mode='lines+markers',
                    name=f'id: {packets["id"][i]}, time tick: {packets["time_tick"][i]}'
                )
            )

        fig.add_annotation(
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            text=f'Sampling rate: {sampling_rate} GHz<br>Offset: {packets["offset_buff"][0] * sampling_interval * 4} ns<br>Threshold: {packets["threshold_buff"][0]}',
            showarrow=False,
            font_size=20,
            align="left",
        )
        fig.update_layout(title='Packet Data Plot', xaxis_title='Time (ns)', yaxis_title='Value')
        fig.show()
