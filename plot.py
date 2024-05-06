import struct
import numpy as np
import plotly.graph_objects as go


class Packets:
    def __init__(self):
        self.length_buff = 0
        self.offset_buff = 0
        self.threshold_buff = 0
        self.ids = []
        self.time_ticks = []
        self.packets = []


def read_packets(file_path, start_offset):
    packets = Packets()

    with open(file_path, 'rb') as file:
        header_offset = start_offset
        data_count = 0
        while True:
            # Read header
            header_length = 16
            file.seek(header_offset)
            header = file.read(header_length)
            if not header:
                break
            header_unpacked = struct.unpack('>8H', header)

            # Output header info
            length_offset = header_unpacked[3]
            packets.offset_buff = length_offset & 0xFF
            packets.length_buff = (length_offset >> 8) & 0xFF
            packets.threshold_buff = header_unpacked[2]

            header_str = ""
            for number in header_unpacked:
                header_str += f"{number:04x} "
            print("Header:", header_str)

            # Read data
            data_length = packets.length_buff * 8
            data_offset = header_offset + header_length
            file.seek(data_offset)
            data = file.read(data_length)
            if not data:
                break
            num_integers = len(data) // 2
            integers = np.asarray(struct.unpack(f'>{num_integers}H', data))
            
            # Output data
            packets.ids.append(header_unpacked[1])
            packets.time_ticks.append(header_unpacked[7])
            packets.packets.append(integers)

            # Move to the next packet
            header_offset = data_offset + data_length
           
            data_count += 1
            if data_count > 5:  # For testing, read only 10 packets
                break
    return packets

# Usage
file_path = 'rx_test1.bin'
start_offset = 104
sampling_interval = 1 # unit ns
sampling_rate = 1 / sampling_interval # unit GHz
packets = read_packets(file_path, start_offset)

# Plotting each packet data as a separate trace
fig = go.Figure()
for i, packet in enumerate(packets.packets):
    fig.add_trace(
        go.Scatter(
            y=packet,
            mode='lines+markers',
            name=f'id: {packets.ids[i]}, time tick: {packets.time_ticks[i]}'
        )
    )

fig.add_annotation (
    x=0.01, y=0.99,
    xref="paper", yref="paper",
    text=f"Sampling rate: {sampling_rate} GHz<br>Offset: {packets.offset_buff * sampling_interval * 4} ns<br>Threshold: {packets.threshold_buff}",
    showarrow=False,
    font_size=20,
    align="left",
)

fig.update_layout(title='Packet Data Plot', xaxis_title='Time (ns)', yaxis_title='Value')
fig.show()

