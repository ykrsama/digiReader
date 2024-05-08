#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import struct
import numpy as np
import plotly.graph_objects as go
import ROOT
from tools import read_bin, plot

MAX_DATA_VALUE = 16384
HEADER_LENGTH = 16
HEADER_FIND_STEP = 8
COLOUR= {
    "FAIL": '\033[91m',
    'OKGREEN': '\033[92m',
    'OKBLUE': '\033[94m',
}

def get_args():
    parser = ArgumentParser(description='Digital Board Reader')
    parser.add_argument('filename', type=Path, help='Input binary file')
    parser.add_argument('-n', '--noroot', action='store_true', help='Do not output root file')
    parser.add_argument('-w', '--wave', default=0, help='Show some waveforms. Won\'t output root file.')
    parser.add_argument('-a', '--algo',
                        default=[],
                        nargs='+',
                        choices=['denoise', 'gmm', 'landau'],
                        help='Apply algorithms')
    parser.add_argument('-i', '--id', nargs='+', type=int, default=[1], help='id begin, id size')
    return parser.parse_args()


def read_packets(file_path, begin_id, end_id, cfgs):
    packets = {key: [] for key in ["length_buff", "data_length", "offset_buff", "threshold_buff", "channel_n",
                                   "id", "time_tick",
                                   "data", "waveform_denoised",
                                   "mean", "std", "max", "min",
                                   "baseline_median", "net_signal_median", "net_signal_denoised_median",
                                   "baseline_landau", "net_signal_landau", "net_signal_denoised_landau",
                                   "baseline_gmm", "net_signal_gmm", "net_signal_denoised_gmm"
                                   ]}
    prev_header_valid = True

    with open(file_path, 'rb') as file:
        file_size = file_path.stat().st_size
        if not begin_id:
            begin_id = 1
        # 1. Find header, starting from offset=0 with id=1, to read the header pattern
        header_offset, header_info = read_bin.find_header_info(file, file_size, 0, 1,
                                                               None, HEADER_LENGTH, HEADER_FIND_STEP)
        header_pattern = (header_info["threshold_buff"], header_info["length_offset"], header_info["channel_n"])
        # NOTE: header_pattern is a fixed subset.
        #       Remember to change header_patter_pos in process_data
        # 2. Header finding (forward direction)
        print(COLOUR["OKBLUE"], end="")
        header_offset, header_info = read_bin.find_header_info(file, file_size, header_offset, begin_id,
                                                               header_pattern, HEADER_LENGTH, HEADER_FIND_STEP)
        # print(f'{COLOUR["OKGREEN"]}Found header id {begin_id} at {hex(header_offset)} : {header_info["str"]}')
        pbar_total = file_size - header_offset
        if end_id:
            pbar_total = (end_id - begin_id + 1) * (header_info["length_buff"] * 4 * 2 + HEADER_LENGTH)
        print(COLOUR["OKGREEN"], end="")
        with tqdm(total=pbar_total, unit='B', unit_scale=True, desc='Reading') as pbar:
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
                        print(COLOUR["FAIL"] + str(e), hex(header_offset), ":", header_info["str"])
                        prev_header_valid = False
                    header_offset += HEADER_FIND_STEP
                    pbar.update(HEADER_FIND_STEP)
                    continue

                if not prev_header_valid:
                    print(COLOUR["OKGREEN"] + 'Find header at', hex(header_offset), ":", header_info["str"])
                    prev_header_valid = True

                if cfgs["wave"]:
                    if len(packets["id"]) >= cfgs["wave"]:
                        break

                if end_id and (header_info["id"] > end_id):
                    break

                # Process Data
                try:
                    result, data_length, data_end_offset = read_bin.process_data(file, data_offset, data_length, header_pattern, cfgs)
                except IOError:
                    break
                except ValueError as e:
                    print(COLOUR["FAIL"] + hex(data_offset), ":", e)
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
        "wave": 0
    }
    start_offset = 0
    sampling_interval = 1 # unit ns
    sampling_rate = 1 / sampling_interval  # unit GHz
    # Read Arguments
    args = get_args()
    file_path = args.filename
    no_root = args.noroot
    cfgs["wave"] = int(args.wave)
    cfgs["algo"] = args.algo
    ids = args.id
    # Vars depend on args
    begin_id = ids[0]
    id_size = ids[1] if len(ids) > 1 else None
    end_id = begin_id + id_size - 1 if id_size else None
    if cfgs["wave"]:
        id_size = cfgs["wave"]
        end_id = begin_id + id_size - 1
    # ===================================================
    # Process binary file
    # ---------------------------------------------------
    packets = read_packets(file_path, begin_id, end_id, cfgs)

    # ===================================================
    # Plotting
    # ---------------------------------------------------
    # Plotting each packet data as a separate trace
    if cfgs["wave"]:
        fig_overlay_wave = go.Figure()
        baseline_algos = ["median", "gmm", "landau"]
        algo_name_map = {
            "median": "Median Baseline",
            "gmm": "GMM Baseline",
            "landau": "Landau Baseline",
        }
        algo_color_map = {
            "median": "green",
            "gmm": "goldenrod",
            "landau": "brown",
        }
        algo_xshift_map = {
            "median": -500,
            "gmm": -250,
            "landau": 0,
        }

        max_range = np.array([float('inf'),-float('inf')])
        for i in np.arange(len(packets["data"])):
            packet_data = packets["data"][i]
            # Plot individual packet
            fig_algo = go.Figure()
            fig_algo.add_trace(go.Scatter(y=packet_data, mode='lines', name='Original Waveform'))
            if 'denoise' in cfgs["algo"]:
                fig_algo.add_trace(go.Scatter(y=packets["waveform_denoised"][i], mode='lines', name='Denoised Waveform',line=dict(dash='dash')))
            for algo in baseline_algos:
                if algo in cfgs["algo"] or algo == "median":
                    fig_algo.add_hline(y=packets["baseline_" + algo][i],line=dict(color=algo_color_map[algo], dash='dashdot'))
                    # Add a dummy trace for legend entry
                    fig_algo.add_trace(go.Scatter(
                        x=[None],
                        y=[None],
                        mode='lines',
                        line=dict(dash='dashdot', color=algo_color_map[algo]),
                        showlegend=True,
                        name=algo_name_map[algo] + (f' ∫V<sub>D</sub> = {round(packets["net_signal_denoised_" + algo][i],2)}' if 'denoise' in cfgs["algo"] else f' ∫V = {round(packets["net_signal_" + algo][i],2)}'),
                    ))

            dy = np.max(packet_data) - np.min(packet_data)
            plot.plot_style(
                fig_algo,
                fontsize2=14,
                yaxis_range=[np.min(packet_data) - 0.03 * dy, np.max(packet_data) + 0.3 * dy],
                title='Digital Board Waveform', xaxis_title='Time (ns)', yaxis_title='Amplitude',
                darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                                 f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                                 f'Threshold {packets["threshold_buff"][0]}'
            )
            fig_algo.show()

            # # overlay plot
            # fig_overlay_wave.add_trace(
            #     go.Scatter(
            #         y=packet_data,
            #         mode='lines+markers',
            #         name=f'id {packets["id"][i]}, time tick {packets["time_tick"][i]}'
            #     )
            # )
            # max_range[0] = np.min(packet_data) if np.min(packet_data) < max_range[0] else max_range[0]
            # max_range[1] = np.max(packet_data) if np.max(packet_data) > max_range[1] else max_range[1]

        # dy = max_range[1] - max_range[0]
        # plot.plot_style(
        #     fig_overlay_wave,
        #     fontsize2=14,
        #     yaxis_range=max_range + dy * np.array([-0.03, 0.3]),
        #     title='Digital Board Waveform', xaxis_title='Time (ns)', yaxis_title='Amplitude',
        #     darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
        #                      f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
        #                      f'Threshold {packets["threshold_buff"][0]}'
        # )
        #
        # fig_overlay_wave.show()

    # ===================================================
    # Output root file
    # ---------------------------------------------------
    if not no_root and not cfgs["wave"]:
        root_file_path = file_path.with_suffix('.root')

        if begin_id:
            root_file_path = root_file_path.with_name(root_file_path.stem + f'.id_{begin_id}_.root')
            if end_id:
                root_file_path = root_file_path.with_name(root_file_path.stem + f'{end_id}.root')

        print(f'{COLOUR["OKBLUE"]}Creating root file: {root_file_path}')
        # Create RDataFrame and Write to file
        no_save = ['data', 'waveform_denoised']
        df = ROOT.RDF.FromNumpy({
            key: np.asarray(value) for key, value in packets.items() if key not in no_save and value
        })
        df.Snapshot("tree", str(root_file_path))
