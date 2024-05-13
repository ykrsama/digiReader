#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ROOT
from astropy.timeseries import LombScargle
from tools import read_bin, plot

MAX_DATA_VALUE = 16384
HEADER_LENGTH = 16
HEADER_FIND_STEP = 8
COLOUR= {
    "FAIL": '\033[91m',
    'OKGREEN': '\033[92m',
    'OKBLUE': '\033[94m',
    'WARNING': '\033[93m',
    'BOLD': '\033[1m',
    'RESET': '\033[0m',
}


def get_args():
    parser = ArgumentParser(description='Digital Board Reader')
    parser.add_argument('filenames', nargs='+', type=Path, help='Input binary file')
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
                                   "id", "timestamp",
                                   "data", "waveform_denoised",
                                   "mean", "std", "max", "min",
                                   "baseline_median", "net_signal_median", "net_signal_denoised_median",
                                   "baseline_landau", "net_signal_landau", "net_signal_denoised_landau",
                                   "baseline_gmm", "net_signal_gmm", "net_signal_denoised_gmm"
                                   ]}
    prev_header_valid = True

    print(COLOUR["OKBLUE"] + 'Opening file:', file_path)
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
        try:
            header_offset, header_info = read_bin.find_header_info(file, file_size, header_offset, begin_id,
                                                               header_pattern, HEADER_LENGTH, HEADER_FIND_STEP)
        except ValueError as e:
            print(f'{COLOUR["FAIL"]}[Error]', e)
            exit()
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
    filenames = args.filenames
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
    show_fig=True
    if len(filenames) > 1 and cfgs["wave"]:
        show_fig = False
        print(f'{COLOUR["WARNING"]+COLOUR["BOLD"]}More than one input file, suppress plot showing.{COLOUR["BOLD"]}')

    for file_path in filenames:
        # ===================================================
        # Process binary file
        # ---------------------------------------------------
        packets = read_packets(file_path, begin_id, end_id, cfgs)

        # ===================================================
        # Plotting
        # ---------------------------------------------------
        # Plotting each packet data as a separate trace
        if cfgs["wave"]:
            label = (f'Sampling rate {sampling_rate} GHz<br>' +
                     f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                     f'Threshold {packets["threshold_buff"][0]}')
            plot_dir = Path('plot')
            print(f'Plotting to {plot_dir}')

            if len(packets["data"]) < 10:
                # Plot individual waveforms
                for i in np.arange(len(packets["data"])):
                    fig = plot.plot_individual_waveform(
                        packets, i,
                        file_path=file_path,
                        label=label,
                        cfgs=cfgs,
                    )
                    plot.write_plot(fig, plot_dir)
                    if show_fig:
                        fig.show()
            else:
                # Plot Waveform Density
                concatenate_time_0 = np.array([])
                concatenate_time = np.array([])
                concatenate_data = np.array([])

                for i in np.arange(len(packets["data"])):
                    packet_data = packets["data"][i]
                    concatenate_time_0 = np.append(concatenate_time_0, np.arange(len(packet_data)))
                    concatenate_time = np.append(concatenate_time, np.arange(len(packet_data)) + 10 * (packets["timestamp"][i] - packets["timestamp"][0]))
                    concatenate_data = np.append(concatenate_data, packet_data)

                for x, title, vline in ((concatenate_time_0, f'Overlay Waveform - {file_path}', packets["offset_buff"][0] * 4), (concatenate_time, f'Waveform - {file_path}', None)):
                    fig = plot.plot_waveform_density(
                        x, concatenate_data,
                        title=title,
                        label=label,
                        vline=vline,
                    )
                    plot.write_plot(fig, plot_dir)
                    if show_fig:
                        fig.show()

                # Compute the Lomb-Scargle periodogram
                frequency, power = LombScargle(concatenate_time * sampling_interval / 1e9, concatenate_data).autopower()
                fig = go.Figure(data=go.Scatter(x=frequency, y=power, mode='lines'))
                dy = np.max(power) - np.min(power)
                plot.plot_style(
                    fig,
                    title=f'LSP - {file_path}',
                    xaxis_title='Frequency [Hz]',
                    yaxis_title='Lomb-Scargle Power',
                    xaxis_type='log',
                    darkshine_label2=label,
                    yaxis_range=[np.min(power) - 0.03 * dy, np.max(power) + 0.3 * dy]
                )
                plot.write_plot(fig, plot_dir)
                if show_fig:
                    fig.show()

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
