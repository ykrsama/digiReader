#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
        if len(packets["data"]) < 10:
            # Plot individual waveforms
            for i in np.arange(len(packets["data"])):
                # Plot individual packet
                packet_data = packets["data"][i]
                fig_algo = go.Figure()
                fig_algo.add_trace(go.Scatter(y=packet_data, mode='lines',
                                              name=f'Original Waveform, <i>I&#x0305;</i>={round(packets["mean"][i], 2)}, <i>σ</i>={round(packets["std"][i], 2)}'))
                if 'denoise' in cfgs["algo"]:
                    fig_algo.add_trace(go.Scatter(y=packets["waveform_denoised"][i], mode='lines', name='Denoised Waveform',
                                                  line=dict(dash='dash')))
                for algo in baseline_algos:
                    if algo in cfgs["algo"] or algo == "median":
                        fig_algo.add_hline(y=packets["baseline_" + algo][i],
                                           line=dict(color=algo_color_map[algo], dash='dashdot'))
                        # Add a dummy trace for legend entry
                        if 'denoise' in cfgs["algo"]:
                            append_name = f' {round(packets["baseline_" + algo][i], 2)}, <i>∫I<sub>D</sub></i> = {round(packets["net_signal_denoised_" + algo][i], 2)}'
                        else:
                            append_name = f' {round(packets["baseline_" + algo][i], 2)}, <i>∫I</i> = {round(packets["net_signal_" + algo][i], 2)}'
                        fig_algo.add_trace(go.Scatter(
                            x=[None],
                            y=[None],
                            mode='lines',
                            line=dict(dash='dashdot', color=algo_color_map[algo]),
                            showlegend=True,
                            name=algo_name_map[algo] + append_name,
                        ))
                dy = np.max(packet_data) - np.min(packet_data)
                plot.plot_style(
                    fig_algo,
                    fontsize2=14,
                    yaxis_range=[np.min(packet_data) - 0.03 * dy, np.max(packet_data) + 0.3 * dy],
                    title='Digital Board Waveform', xaxis_title='Time (ns)', yaxis_title='Intensity',
                    darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                                     f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                                     f'Threshold {packets["threshold_buff"][0]}'
                )
                fig_algo.show()
        else:
            concatenate_time_0 = np.array([])
            concatenate_time = np.array([])
            concatenate_data = np.array([])
            for i in np.arange(len(packets["data"])):
                packet_data = packets["data"][i]
                concatenate_time_0 = np.append(concatenate_time_0, np.arange(len(packet_data)))
                concatenate_time = np.append(concatenate_time, np.arange(len(packet_data)) + 10 * (packets["time_tick"][i] - packets["time_tick"][0]))
                concatenate_data = np.append(concatenate_data, packet_data)

            for x, title in ((concatenate_time_0, 'Overlay Digital Board Waveform'), (concatenate_time, 'Digital Board Waveform')):
                fig = px.density_heatmap(
                    x=x, y=concatenate_data,
                    marginal_x="histogram",
                    marginal_y="histogram",
                    nbinsx=min([200, int(x.max() - x.min() + 1)]),
                )
                # fig.update_layout(coloraxis_showscale=False)
                plot.plot_style(
                    fig,
                    fontsize2=14,
                    width=1200, height=800,
                    title=title, xaxis_title='Time (ns)', yaxis_title='Intensity',
                    darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                                     f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                                     f'Threshold {packets["threshold_buff"][0]}'
                )
                fig.update_xaxes(title_text="", row=1, col=2)  # Marginal y
                fig.update_xaxes(title_text="", row=2, col=1)  # Marginal x
                fig.update_yaxes(title_text="", row=2, col=1)  # Marginal x
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
