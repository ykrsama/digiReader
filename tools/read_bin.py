import struct
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
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
        "id": header_unpacked[0] * 0x10000 + header_unpacked[1],
        "threshold_buff": header_unpacked[2],
        "length_offset": length_offset,
        "offset_buff": length_offset & 0xFF,
        "length_buff": (length_offset >> 8) & 0xFF,
        "channel_n": header_unpacked[4],
        "time_tick": header_unpacked[5] * 0x100000000 + header_unpacked[6] * 0x10000 + header_unpacked[7],
        "str": ints_to_str(header_unpacked)
    }
    return header_info, header_unpacked


def check_header(header_unpacked: tuple, header_pattern: tuple, data_max=16384):
    if header_unpacked[2:5] != header_pattern:
        raise ValueError(f'Invalid header {header_unpacked} ~ {header_pattern}')


def check_data(data_unpacked, data_max=16384):
    if max(data_unpacked) > data_max:
        raise ValueError(f'Invalid data {hex(max(data_unpacked))} > {data_max}')



def is_begin_header(header_unpacked: tuple, target_id=1, data_max=16384):
        return (header_unpacked[0] * 0x10000 + header_unpacked[1] == target_id and
                max(header_unpacked) > data_max)


def find_header_info(file, filesize, start_offset, target_id: int, header_pattern=None, header_length=16, find_step=8):
    header_offset = start_offset
    finding_direction = 1  # 1: forward finding, -1: backward finding
    finding_direction_turned = 0
    if header_pattern:
        with tqdm(total=filesize, initial=header_offset, unit='B', unit_scale=True, desc=f'Finding Header id {target_id}') as pbar:
            while True:
                header = get_byte(file, header_offset, header_length)
                header_info, header_unpacked = byte_to_header_info(header)
                try:
                    check_header(header_unpacked, header_pattern)
                    if header_info["id"] == target_id:
                        return header_offset, header_info
                    else:
                        if (target_id - header_info["id"]) * finding_direction < 0:
                            # Change direction
                            finding_direction_turned += 1
                            finding_direction = -1 if finding_direction > 0 else 1
                        else:
                            finding_direction = 2 * (target_id - header_info["id"])
                except ValueError:
                    finding_direction = 1 if finding_direction > 0 else -1
                if finding_direction_turned > 3:
                    raise ValueError(f'Header Not Found')
                # Move to next header
                if header_offset + finding_direction * find_step < 0:
                    # If next header < 0, reverse direction
                    finding_direction_turned += 1
                    pbar.update(0 - header_offset)
                    header_offset = 0
                    finding_direction = 1
                elif header_offset + finding_direction * find_step > filesize - header_length:
                    # Exceeded
                    finding_direction_turned += 1
                    jump_to = (filesize // find_step) * find_step
                    pbar.update(jump_to - header_offset)
                    header_offset = jump_to - header_length
                    finding_direction = -1
                else:
                    pbar.update(finding_direction * find_step)
                    header_offset += finding_direction * find_step

    else:
        while True:
            header = get_byte(file, header_offset, header_length)
            header_info, header_unpacked = byte_to_header_info(header)
            if is_begin_header(header_unpacked, target_id):
                return header_offset, header_info
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




def process_header(file, header_offset: int, header_length: int, header_pattern: tuple):
    # Read header
    header = get_byte(file, header_offset, header_length)
    header_info, header_unpacked = byte_to_header_info(header)
    # Check header
    check_header(header_unpacked, header_pattern)
    # For Data offset
    data_offset = header_offset + header_length
    data_length = header_info["length_buff"] * 4 * 2
    return header_info, data_offset, data_length


def process_data(file, data_offset: int, data_length: int, header_pattern: tuple, cfgs: dict, header_pattern_pos=2):
    data = get_byte(file, data_offset, data_length)
    data_unpacked = np.asarray(byte_to_int(data))
    # Check data: if data length is shorter than expected length_buff, mixing next header in data
    header_pos = find_sublist(data_unpacked, np.asarray(header_pattern))
    if header_pos >= 0:
        # Found header in data
        if header_pos > header_pattern_pos:
            data_num = header_pos - header_pattern_pos
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
    if 'denoise' in cfgs["algo"]:
        waveform_denoised = savgol_filter(data_unpacked,
                                          window_length=cfgs["denoise_savgol_window_length"],
                                          polyorder=cfgs["denoise_savgol_polyorder"])
    else:
        waveform_denoised = data_unpacked
    # Estimate baseline using median
    baseline_median = np.median(waveform_denoised)
    result["baseline_median"] = baseline_median
    result["net_signal_median"] = np.sum(data_unpacked - baseline_median)
    if 'denoise' in cfgs["algo"]:
        result["net_signal_denoised_median"] = np.sum(waveform_denoised - baseline_median)
    # Estimate baseline using Gaussian Mixture
    if 'gmm' in cfgs["algo"]:
        baseline_gmm = estimate_baseline_gmm(waveform_denoised, cfgs["gmm_n_components"])
        result["baseline_gmm"] = baseline_gmm
        result["net_signal_gmm"] = np.sum(data_unpacked - baseline_gmm)
        if 'denoise' in cfgs["algo"]:
            result["net_signal_denoised_gmm"] = np.sum(waveform_denoised - baseline_gmm)
    # Estimate baseline using Landau distribution fit
    if 'landau' in cfgs["algo"]:
        baseline_landau = estimate_baseline_landau(waveform_denoised)
        result["baseline_landau"] = baseline_landau
        result["net_signal_landau"] = np.sum(data_unpacked - baseline_landau)
        if 'denoise' in cfgs["algo"]:
            result["net_signal_denoised_landau"] = np.sum(waveform_denoised - baseline_landau)
    if cfgs["wave"]:
        result["data"] = data_unpacked
        if 'denoise' in cfgs["algo"]:
            result["waveform_denoised"] = waveform_denoised
    return result, data_length, data_end_offset
