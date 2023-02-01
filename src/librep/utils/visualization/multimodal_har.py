import numpy as np
import plotly.graph_objs as go
from typing import List
from librep.datasets.multimodal.multimodal import MultiModalDataset


def plot_windows_sample(
    dataset: MultiModalDataset,
    windows: List[str] = None,
    sample_idx: int = 0,
    the_slice: slice = slice(None, None, None),
    title: str = "",
    mode: str = "lines",
    showlegend: bool = True,
    xaxis: str = "x",
    yaxis: str = "y",
    return_traces_layout: bool = False
):
    if windows is None:
        windows = dataset.window_names

    traces = [
        go.Scatter(y=dataset.windows(window)[sample_idx][0][the_slice], name=window, mode=mode)
        for window in windows
    ]
    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
    )

    if return_traces_layout:
        return traces, layout
    else:
        return go.Figure(data=traces, layout=layout)


def plot_twin_windows_sample(
    dataset: MultiModalDataset,
    sample_idx: int = 0,
    y1_windows: List[str] = ("accel-x", "accel-y", "accel-z"),
    y2_windows: List[str] = ("gyro-x", "gyro-y", "gyro-z"),
    title: str = "",
    mode: str = "lines",
    showlegend: bool = True,
    xaxis: str = "x",
    y1_axis: str = "y1",
    y2_axis: str = "y2",
    the_slice: slice = slice(None, None, None),
    return_traces_layout: bool = False
):
    start = 0
    traces = []
    for window in y1_windows:
        window_val = dataset.windows(window)[sample_idx][0][the_slice]
        traces.append(
            go.Scatter(
                x=np.arange(start, start + window_val.shape[0]),
                y=window_val,
                name=window,
                mode=mode,
                yaxis="y1"
            )
        )
        start += window_val.shape[0]

    start = 0
    for window in y2_windows:
        window_val = dataset.windows(window)[sample_idx][0][the_slice]
        traces.append(
            go.Scatter(
                x=np.arange(start, start + window_val.shape[0]),
                y=window_val,
                name=window,
                mode=mode,
                yaxis="y2"
            )
        )
        start += window_val.shape[0]

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        xaxis=dict(title=xaxis),
        yaxis=dict(
            title=y1_axis,
            side='left',
        ),
        yaxis2=dict(
            title=y2_axis,
            side='right',
            overlaying='y'
        )
    )

    if return_traces_layout:
        return traces, layout
    else:
        return go.Figure(data=traces, layout=layout)

def plot_windows_single_line(
    dataset: MultiModalDataset,
    windows: List[str],
    sample_idx: int = 0,
    the_slice: slice = slice(None, None, None),
    title: str = "",
    mode: str = "lines",
    showlegend: bool = True,
    xaxis: str = "x",
    yaxis: str = "y",
    return_traces_layout: bool = False
):
    if windows is None:
            windows = dataset.window_names
    traces = []
    start = 0
    for window in windows:
        window_val = dataset.windows(window)[sample_idx][0][the_slice]
        traces.append(
            go.Scatter(
                x=np.arange(start, start + window_val.shape[0]),
                y=window_val,
                name=window,
                mode=mode,
            )
        )
        start += window_val.shape[0]

    layout = go.Layout(
        title=title,
        showlegend=showlegend,
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
    )

    if return_traces_layout:
        return traces, layout
    else:
        return go.Figure(data=traces, layout=layout)