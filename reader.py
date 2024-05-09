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
PLOT_FORMATS=('png', 'pdf', 'json')

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
                                   "id", "time_tick",
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
            plot_dir = file_path.parent / 'plot'
            print(f'Plotting to {plot_dir}')
            for format in PLOT_FORMATS:
                Path.mkdir(plot_dir / format, parents=True, exist_ok=True)
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
            if len(packets["data"]) < 10:
                # ===================================================
                # Plot individual waveforms
                # ---------------------------------------------------
                for i in np.arange(len(packets["data"])):
                    # Plot individual packet
                    packet_data = packets["data"][i]
                    fig_algo = go.Figure()
                    fig_algo.add_trace(
                        go.Scatter(
                            y=packet_data, mode='lines',
                            name=f'Original Waveform'  # ' <i>v&#x0305;</i>={round(packets["mean"][i], 2)}, <i>σ</i>={round(packets["std"][i], 2)}'
                        )
                    )
                    if 'denoise' in cfgs["algo"]:
                        waveform_denoised = packets["waveform_denoised"][i]
                        fig_algo.add_trace(
                            go.Scatter(
                                y=waveform_denoised,
                                mode='lines',
                                name=f'Denoised Waveform', # ', <i>v&#x0305;</i>={round(np.mean(waveform_denoised))}, <i>σ</i>={round(np.std(waveform_denoised),2)}',
                                line=dict(dash='dash')
                            )
                        )
                    for algo in baseline_algos:
                        if algo in cfgs["algo"] or algo == "median":
                            fig_algo.add_hline(y=packets["baseline_" + algo][i],
                                               line=dict(color=algo_color_map[algo], dash='dashdot'))
                            # Add a dummy trace for legend entry
                            if 'denoise' in cfgs["algo"]:
                                append_name = f' {round(packets["baseline_" + algo][i], 2)}, <i>∫v<sub>D</sub></i> = {round(packets["net_signal_denoised_" + algo][i], 2)}'
                            else:
                                append_name = f' {round(packets["baseline_" + algo][i], 2)}, <i>∫v</i> = {round(packets["net_signal_" + algo][i], 2)}'
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
                        fontsize2=14, label_font=14,
                        yaxis_range=[np.min(packet_data) - 0.03 * dy, np.max(packet_data) + 0.3 * dy],
                        title=f'ID {packets["id"][i]} - {file_path}', xaxis_title='Time (ns)', yaxis_title='Amplitude',
                        darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                                         f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                                         f'Threshold {packets["threshold_buff"][0]}',
                    )
                    fig_algo.show()
            else:
                # ===================================================
                # Plot Waveform Histo
                # ---------------------------------------------------
                concatenate_time_0 = np.array([])
                concatenate_time = np.array([])
                concatenate_data = np.array([])
                for i in np.arange(len(packets["data"])):
                    packet_data = packets["data"][i]
                    concatenate_time_0 = np.append(concatenate_time_0, np.arange(len(packet_data)))
                    concatenate_time = np.append(concatenate_time, np.arange(len(packet_data)) + 10 * (packets["time_tick"][i] - packets["time_tick"][0]))
                    concatenate_data = np.append(concatenate_data, packet_data)

                for x, title in ((concatenate_time_0, f'Overlay Waveform - {file_path}'), (concatenate_time, f'Waveform - {file_path}')):
                    fig = px.density_heatmap(
                        x=x, y=concatenate_data,
                        marginal_x="histogram",
                        marginal_y="histogram",
                        nbinsx=min([200, int(x.max() - x.min() + 1)]),
                    )
                    # fig.update_layout(coloraxis_showscale=False)
                    fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(size=16)))
                    plot.plot_style(
                        fig,
                        label_font=20,
                        width=1200, height=800,
                        title=title, xaxis_title='Time (ns)', yaxis_title='Amplitude',
                        darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                                         f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                                         f'Threshold {packets["threshold_buff"][0]}',
                        darkshine_label_shift=[0.72, -0.01],
                    )
                    fig.update_xaxes(title_text="", row=1, col=2)  # Marginal y
                    fig.update_xaxes(title_text="", row=2, col=1)  # Marginal x
                    fig.update_yaxes(title_text="Sample Count", row=2, col=1)  # Marginal x
                    fig.add_annotation(
                        x=0.96, y=-0.09,
                        xref="paper", yref="paper",
                        text="Amplitude Count",
                        font_size=20,
                        showarrow=False
                    )
                    # Show mean and std of total waveform
                    fig.add_annotation(
                        x=0.99, y=0.72,
                        xanchor='right', yanchor='top',
                        xref="paper", yref="paper",
                        text=f'Mean {round(np.mean(concatenate_data),2)}<br>Std Dev {round(np.std(concatenate_data),2)}',
                        font_size=18,
                        showarrow=False,
                        align="left",
                        bordercolor="#666666",
                        borderwidth=2,
                        borderpad=4,
                    )
                    fig.show()
                    for format in PLOT_FORMATS:
                        fig.write_image(plot_dir / format / Path(plot.sanitize_filename(title) + '.' + format), scale=2 if format=='png' else 1)

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
