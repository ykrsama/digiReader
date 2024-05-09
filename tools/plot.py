import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np

marker_symbol_cycle = ['circle', 'square', 'cross', 'star-square', 'star-diamond', 'pentagon',
                       'diamond-tall', 'diamond-wide', 'star', 'hourglass']

def plot_style(fig, title=None,
               width=800, height=600,
               xaxis_title=None, yaxis_title=None,
               xaxis_type=None, yaxis_type=None,
               xaxis_tickangle=None,
               xaxis_range=None, yaxis_range=None,
               fontsize1=20, fontsize2=16, label_font=16,
               darkshine_label1=r'<b><i>Dark SHINE</i></b>',
               darkshine_label2=r'1×10<sup>14</sup> Events @ 8 GeV',
               darkshine_label3=None,
               darkshine_label_shift=[0,0]):
    axis_attr = dict(
        linecolor="#666666",
        gridcolor='#F0F0F0',
        zerolinecolor='rgba(0,0,0,0)',
        linewidth=2,
        gridwidth=1,
        showline=True,
        showgrid=False,
        mirror=True,
        tickfont=dict(size=16),
    )
    fig.update_xaxes(
        **axis_attr,
        title=dict(
            text=xaxis_title,
            font_size=fontsize1
        ),
        range=xaxis_range,
        type=xaxis_type,
        tickangle=xaxis_tickangle,
    )
    fig.update_yaxes(
        **axis_attr,
        showexponent='all',
        exponentformat='power',
        type=yaxis_type,
        title=dict(
            text=yaxis_title,
            font_size=fontsize1
        ),
        range=yaxis_range
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font_size=fontsize1
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',  # Anchor point of the legend ('left' or 'right')
            yanchor='top',  # Anchor point of the legend ('top' or 'bottom')
            bgcolor='rgba(0,0,0,0)',
            font_size=fontsize2,
            groupclick="toggleitem",
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(
            l=90,
            r=20,
            b=20,
            t=80,
        ),
        width=width,
        height=height,
    )
    # Rearrange the legend items to form two columns
    n = len(fig.data)  # Number of traces
    if n > 3:
        fig.update_layout(
            legend=dict(orientation="h")
        )
        nhalf = round(n/2)
        for i in range(nhalf):
            fig.data[i].legendgroup = 'group2'
        for i in range(nhalf, n):
                fig.data[i].legendgroup = 'group1'
    if darkshine_label1:
        fig.add_annotation(x=0.04 + darkshine_label_shift[0], y=0.98 + darkshine_label_shift[1],
                           xanchor='left', yanchor='top',
                           xref="paper", yref="paper",
                           text=darkshine_label1,
                           font_size=label_font + 2,
                           showarrow=False,
                           align="left",
                           )
    if darkshine_label2:
        fig.add_annotation(x=0.04 + darkshine_label_shift[0], y=0.93 + darkshine_label_shift[1],
                           xanchor='left', yanchor='top',
                           xref="paper", yref="paper",
                           text=darkshine_label2,
                           font_size=label_font,
                           showarrow=False,
                           align="left",
                           )
    if darkshine_label3:
        fig.add_annotation(x=0.04 + darkshine_label_shift[0], y=0.88 + darkshine_label_shift[1],
                           xref="paper", yref="paper",
                           text=darkshine_label3,
                           font_size=label_font,
                           showarrow=False,
                           align="left",
                           )


def sanitize_filename(filename):
    # Define illegal characters that are typically not allowed in file names
    filename = filename.replace(' - ', '_')
    illegal_chars = '<>:"/\\|?* '
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename


def write_plot(fig, plot_dir: Path, plot_formats=('png', 'pdf', 'json')):
    title = fig.layout.title.text
    for format in plot_formats:
        Path.mkdir(plot_dir / format, parents=True, exist_ok=True)
    for format in plot_formats:
        fig.write_image(plot_dir / format / Path(sanitize_filename(title) + '.' + format),
                        scale=2 if format == 'png' else 1)


def plot_individual_waveform(packets, i, file_path, sampling_rate, sampling_interval, cfgs):
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
    # Plot individual packet
    packet_data = packets["data"][i]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=packet_data, mode='lines',
            name=f'Original Waveform'
            # ' <i>v&#x0305;</i>={round(packets["mean"][i], 2)}, <i>σ</i>={round(packets["std"][i], 2)}'
        )
    )
    if 'denoise' in cfgs["algo"]:
        waveform_denoised = packets["waveform_denoised"][i]
        fig.add_trace(
            go.Scatter(
                y=waveform_denoised,
                mode='lines',
                name=f'Denoised Waveform',
                # ', <i>v&#x0305;</i>={round(np.mean(waveform_denoised))}, <i>σ</i>={round(np.std(waveform_denoised),2)}',
                line=dict(dash='dash')
            )
        )
    for algo in baseline_algos:
        if algo in cfgs["algo"] or algo == "median":
            fig.add_hline(y=packets["baseline_" + algo][i],
                               line=dict(color=algo_color_map[algo], dash='dashdot'))
            # Add a dummy trace for legend entry
            if 'denoise' in cfgs["algo"]:
                append_name = f' {round(packets["baseline_" + algo][i], 2)}, <i>∫v<sub>D</sub></i> = {round(packets["net_signal_denoised_" + algo][i], 2)}'
            else:
                append_name = f' {round(packets["baseline_" + algo][i], 2)}, <i>∫v</i> = {round(packets["net_signal_" + algo][i], 2)}'
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(dash='dashdot', color=algo_color_map[algo]),
                showlegend=True,
                name=algo_name_map[algo] + append_name,
            ))
    dy = np.max(packet_data) - np.min(packet_data)
    plot_style(
        fig,
        fontsize2=14, label_font=14,
        yaxis_range=[np.min(packet_data) - 0.03 * dy, np.max(packet_data) + 0.3 * dy],
        title=f'ID {packets["id"][i]} - {file_path}', xaxis_title='Time (ns)', yaxis_title='Amplitude',
        darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                         f'Offset {packets["offset_buff"][0] * sampling_interval * 4} ns<br>' +
                         f'Threshold {packets["threshold_buff"][0]}',
    )
    return fig

def plot_waveform_density(x, y, title, sampling_rate, sampling_interval, offset_buff, threashold_buff, vline=None):
    fig = px.density_heatmap(
        x=x, y=y,
        marginal_x="histogram",
        marginal_y="histogram",
        nbinsx=min([200, int(x.max() - x.min() + 1)]),
    )
    if vline:
        fig.add_vline(
            x=vline,
            line=dict(color='white', dash='dash'),
            row=1, col=1,
        )
    # fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(size=16)))
    plot_style(
        fig,
        label_font=20,
        width=1200, height=800,
        title=title, xaxis_title='Time (ns)', yaxis_title='Amplitude',
        darkshine_label2=f'Sampling rate {sampling_rate} GHz<br>' +
                         f'Offset {offset_buff * sampling_interval * 4} ns<br>' +
                         f'Threshold {threashold_buff}',
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
        text=f'Mean {round(np.mean(y), 2)}<br>Std Dev {round(np.std(y), 2)}',
        font_size=18,
        showarrow=False,
        align="left",
        bordercolor="#666666",
        borderwidth=2,
        borderpad=4,
    )
    return fig