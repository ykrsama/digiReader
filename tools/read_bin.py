import struct
import numpy as np
from scipy.signal import savgol_filter
from .algo import estimate_baseline_gmm, estimate_baseline_landau


def get_byte(file, byte_offset: int, byte_length: int):
    file.seek(byte_offset)
    byte = file.read(byte_length)
    if not byte or len(byte) != byte_length:
        raise IOError('Byte error')
    return byte


def byte_to_int(byte: bytes):
    byte_num = len(byte) // 2
    return struct.unpack(f'>{byte_num}H', byte)


def ints_to_str(ints: tuple):
    ret_str = ""
    for number in ints:
        ret_str += f"{number:04x} "
    return ret_str


def byte_to_header_info(header: bytes):
    header_unpacked = byte_to_int(header)
    length_offset = header_unpacked[3]
    header_info = {
        "id": header_unpacked[1],
        "threshold_buff": header_unpacked[2],
        "length_offset": length_offset,
        "offset_buff": length_offset & 0xFF,
        "length_buff": (length_offset >> 8) & 0xFF,
        "channel_n": header_unpacked[4],
        "time_tick": header_unpacked[7],
        "str": ints_to_str(header_unpacked)
    }
    return header_info, header_unpacked


def is_begin_header(header_info: dict, begin_id=1, buff_info=None, max_channel_n=8):
    if not begin_id:
        begin_id = 1
    if buff_info:
        return (header_info["id"] == begin_id and
            header_info["length_offset"] == buff_info and
            header_info["channel_n"] < max_channel_n)
    else:
        return (header_info["id"] == begin_id and
            header_info["channel_n"] < max_channel_n and
            header_info["offset_buff"] <= header_info["length_buff"])


def find_begin_header_info(file, begin_id: int, buff_info: int, header_length=16, find_step=8):
    header_offset = 0
    while True:
        header = get_byte(file, header_offset, header_length)
        header_info, _ = byte_to_header_info(header)
        if is_begin_header(header_info, begin_id, buff_info):
            return header_offset, header_info
        else:
            header_offset += find_step


def find_sublist(lst: np.ndarray, sublst: np.ndarray):
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


def check_header(header_unpacked: tuple, header_pattern_3_5: tuple):
    if header_unpacked[3:5] != header_pattern_3_5:
        raise ValueError('Invalid header')


def check_data(data_unpacked, data_max=16384):
    if np.max(data_unpacked) > data_max:
        raise ValueError(f'Invalid data {hex(np.max(data_unpacked))} > {data_max}')


def process_header(file, header_offset: int, header_length: int, header_pattern: tuple):
    # Read header
    header = get_byte(file, header_offset, header_length)
    header_info, header_unpacked = byte_to_header_info(header)
    # Check header
    check_header(header_unpacked, header_pattern)
    # For Data offset
    data_offset = header_offset + header_length
    data_length = header_info["length_buff"] * 8
    return header_info, data_offset, data_length


def process_data(file, data_offset: int, data_length: int, header_pattern: tuple, cfgs: dict):
    data = get_byte(file, data_offset, data_length)
    data_unpacked = np.asarray(byte_to_int(data))
    # Check data: if data length is shorter than expected length_buff, mixing next header in data
    header_pos = find_sublist(data_unpacked, np.asarray(header_pattern))
    if header_pos >= 0:
        # Found header in data
        if header_pos > 3:
            data_num = header_pos - 3
            data_length = data_num * 2
            data_unpacked = data_unpacked[:data_num]
        else:
            raise ValueError('No data')
    # Check data: value
    check_data(data_unpacked)
    data_end_offset = data_offset + data_length
    # ===================================================
    # Process Data
    # ---------------------------------------------------
    result = {
        "mean": np.mean(data_unpacked),
        "std": np.std(data_unpacked),
        "max": np.max(data_unpacked),
        "min": np.min(data_unpacked),
    }
    # Denoise the waveform
    if 'denoise' in cfgs["modes"]:
        waveform_denoised = savgol_filter(data_unpacked,
                                          window_length=cfgs["denoise_savgol_window_length"],
                                          polyorder=cfgs["denoise_savgol_polyorder"])
    else:
        waveform_denoised = data_unpacked
    # Estimate baseline using median
    baseline_median = np.median(waveform_denoised)
    result["baseline_median"] = baseline_median
    result["net_signal_median"] = np.sum(data_unpacked - baseline_median)
    if 'denoise' in cfgs["modes"]:
        result["net_signal_denoised_median"] = np.sum(waveform_denoised - baseline_median)
    # Estimate baseline using Gaussian Mixture
    if 'gmm' in cfgs["modes"]:
        baseline_gmm = estimate_baseline_gmm(waveform_denoised, cfgs["gmm_n_components"])
        result["baseline_gmm"] = baseline_gmm
        result["net_signal_gmm"] = np.sum(data_unpacked - baseline_gmm)
        if 'denoise' in cfgs["modes"]:
            result["net_signal_denoised_gmm"] = np.sum(waveform_denoised - baseline_gmm)
    # Estimate baseline using Landau distribution fit
    if 'landau' in cfgs["modes"]:
        baseline_landau = estimate_baseline_landau(waveform_denoised)
        result["baseline_landau"] = baseline_landau
        result["net_signal_landau"] = np.sum(data_unpacked - baseline_landau)
        if 'denoise' in cfgs["modes"]:
            result["net_signal_denoised_landau"] = np.sum(waveform_denoised - baseline_landau)
    if 'wave' in cfgs["modes"]:
        result["data"] = data_unpacked
    return result, data_length, data_end_offset
