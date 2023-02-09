import pandas as pd
import numpy as np
from librep.datasets.multimodal.multimodal import (
    PandasMultiModalDataset,
    concat
)
import pytest

@pytest.fixture
def multimodal_dataframe_1():
    df = pd.DataFrame(np.arange(40).reshape(5, 8), columns=["accel-0", "accel-1", "accel-2", "accel-3", "gyro-0", "gyro-1", "gyro-2", "gyro-3"])
    df["label"] = ["a", "b", "c", "d", "e"]
    return PandasMultiModalDataset(
        df, feature_prefixes=["accel", "gyro"], label_columns="label"
    )

@pytest.fixture
def multimodal_dataframe_2():
    df = pd.DataFrame(np.arange(40).reshape(5, 8)*2, columns=["accel-0", "accel-1", "accel-2", "accel-3", "gyro-0", "gyro-1", "gyro-2", "gyro-3"])
    df["label"] = ["f", "g", "h", "i", "j"]
    return PandasMultiModalDataset(
        df, feature_prefixes=["accel", "gyro"], label_columns="label"
    )

@pytest.fixture
def multimodal_dataframe_mag():
    df = pd.DataFrame(np.arange(30).reshape(5, 6)*-2, columns=["mag-x-0", "mag-x-1", "mag-y-0", "mag-y-1", "mag-z-0", "mag-z-1"])
    df["label"] = ["v", "w", "x", "y", "z"]
    return PandasMultiModalDataset(
        df, feature_prefixes=["mag-x", "mag-y", "mag-z"], label_columns="label"
    )


def test_dataframe_window(multimodal_dataframe_1: PandasMultiModalDataset):
    assert multimodal_dataframe_1.window_names == ["accel", "gyro"]
    assert multimodal_dataframe_1.window_slices == [
        (0, 4), (4, 8)
    ]

    assert np.all(multimodal_dataframe_1[0][0] == np.arange(40).reshape(5, 8)[0, 0:8])
    assert multimodal_dataframe_1.feature_columns == ["accel-0", "accel-1", "accel-2", "accel-3", "gyro-0", "gyro-1", "gyro-2", "gyro-3"]
    assert multimodal_dataframe_1.feature_prefixes == ["accel", "gyro"]
    assert multimodal_dataframe_1.feature_windows[1]["prefix"] == "gyro"
    assert multimodal_dataframe_1.feature_windows[1]["features"] == ["gyro-0", "gyro-1", "gyro-2", "gyro-3"]
    assert multimodal_dataframe_1.feature_windows[1]["start"] == 4
    assert multimodal_dataframe_1.feature_windows[1]["end"] == 8
    assert multimodal_dataframe_1[1][1] == "b"
    assert np.all(multimodal_dataframe_1[1][0] == [8, 9, 10, 11, 12, 13, 14, 15])
    assert multimodal_dataframe_1.windows("gyro")[3][1] == "d"
    assert np.all(multimodal_dataframe_1.windows("gyro")[3][0] == [28, 29, 30, 31])


def test_dataframe_concat_axis_0(multimodal_dataframe_1: PandasMultiModalDataset, multimodal_dataframe_2: PandasMultiModalDataset):
    z = concat((multimodal_dataframe_1, multimodal_dataframe_2), axis=0)
    assert multimodal_dataframe_1.window_names == ["accel", "gyro"]
    assert multimodal_dataframe_1.window_slices == [
        (0, 4), (4, 8)
    ]
    assert z.feature_columns == ["accel-0", "accel-1", "accel-2", "accel-3", "gyro-0", "gyro-1", "gyro-2", "gyro-3"]
    assert z.feature_prefixes == ["accel", "gyro"]
    assert z.feature_windows[1]["prefix"] == "gyro"
    assert z.feature_windows[1]["features"] == ["gyro-0", "gyro-1", "gyro-2", "gyro-3"]
    assert z.feature_windows[1]["start"] == 4
    assert z.feature_windows[1]["end"] == 8
    assert multimodal_dataframe_1.feature_windows[1]["start"] == 4
    assert multimodal_dataframe_1.feature_windows[1]["end"] == 8
    assert z.data.shape == (10, 9)
    assert np.all(z[0][0] == multimodal_dataframe_1[0][0])
    assert np.all(z[5][0] == multimodal_dataframe_2[0][0])
    assert np.all(z[0][1] == multimodal_dataframe_1[0][1])
    assert np.all(z[5][1] == multimodal_dataframe_2[0][1])


def test_dataframe_concat_axis_1(multimodal_dataframe_1: PandasMultiModalDataset, multimodal_dataframe_mag: PandasMultiModalDataset):
    z = concat((multimodal_dataframe_1, multimodal_dataframe_mag), axis=1)
    assert np.all(multimodal_dataframe_1.window_names == ["accel", "gyro"])
    assert np.all(z.window_slices == [
        (0, 4), (4, 8), (9, 11), (11, 13), (13, 15)
    ])
    assert np.all(z.feature_columns == ["accel-0", "accel-1", "accel-2", "accel-3", "gyro-0", "gyro-1", "gyro-2", "gyro-3", "mag-x-0", "mag-x-1", "mag-y-0", "mag-y-1", "mag-z-0", "mag-z-1"])
    assert np.all(z.feature_prefixes == ["accel", "gyro", "mag-x", "mag-y", "mag-z"])
    assert np.all(z.feature_windows[1]["prefix"] == "gyro")
    assert np.all(z.feature_windows[1]["features"] == ["gyro-0", "gyro-1", "gyro-2", "gyro-3"])
    assert z.feature_windows[1]["start"] == 4
    assert z.feature_windows[1]["end"] == 8
    assert np.all(multimodal_dataframe_1.feature_windows[1]["start"] == 4)
    assert np.all(multimodal_dataframe_1.feature_windows[1]["end"] == 8)
    assert z.data.shape == (5, 15)

# def test_array_dataset_window(multimodal_array_1):
#     assert multimodal_array_1.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1"]
#     assert multimodal_array_1.window_slices == [
#         (0, 2),
#         (2, 4),
#         (4, 6),
#         (6, 8),
#     ]
#     assert multimodal_array_1.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1"]
#     assert np.all(multimodal_array_1[0][0] == np.arange(40).reshape(5, 8)[0, 0:8])
#     assert np.all(multimodal_array_1[2][0] == np.arange(40).reshape(5, 8)[2, 0:8])
#     assert np.all(multimodal_array_1.windows("accel-1")[0][0] == np.arange(40).reshape(5, 8)[0, 2:4])
#     assert np.all(multimodal_array_1.windows(["accel-0", "accel-1"])[0][0] == np.arange(40).reshape(5, 8)[0, 0:4])
#     assert np.all(multimodal_array_1.windows(["gyro-0", "gyro-1"])[0][0] == np.arange(40).reshape(5, 8)[0, 4:8])
#     assert np.all(multimodal_array_1.windows(["accel-1", "gyro-1"])[0][0] == np.array([2, 3, 6, 7]))
#     assert np.all(multimodal_array_1.windows(["accel-0", "gyro-0"])[0][0] == np.array([0, 1, 4, 5]))
#     assert multimodal_array_1.windows(["accel-0", "gyro-0"])[0][1] == 0
#     assert multimodal_array_1.windows(["accel-0", "gyro-0"])[1][1] == 1
#     assert multimodal_array_1.windows(["accel-0", "gyro-0"])[2][1] == 2
#     assert multimodal_array_1.windows(["accel-0", "gyro-0"])[3][1] == 3
#     assert multimodal_array_1.windows(["accel-0", "gyro-0"]).y[4] == 4

#     assert multimodal_array_1.windows(["accel-0", "gyro-0"]).window_names == [
#         "accel-0",
#         "gyro-0",
#     ]
#     assert multimodal_array_1.windows(["accel-0", "gyro-0"]).window_slices == [
#         (0, 2),
#         (2, 4),
#     ]
#     assert multimodal_array_1.windows(["accel-1", "gyro-0"]).window_names == [
#         "accel-1",
#         "gyro-0",
#     ]
#     assert multimodal_array_1.windows(["accel-0", "accel-1", "gyro-0"]).window_slices == [
#         (0, 2), (2, 4), (4, 6)
#     ]

#     with pytest.raises(ValueError):
#         multimodal_array_1.windows(["accel-0", "accel-1", "gyro-0", "gyro-1", "mag-0"])
#         multimodal_array_1.windows(["accel"])


# def test_dataset_concatenate(multimodal_array_1, multimodal_array_2):
#     merged = multimodal_array_1.concatenate(multimodal_array_2)
#     assert len(merged) == 10

#     assert merged.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1"]
#     assert merged.window_slices == [(0, 2), (2, 4), (4, 6), (6, 8)]

#     assert merged.windows(["accel-0", "gyro-0"]).window_names == [
#         "accel-0",
#         "gyro-0",
#     ]
#     assert merged.windows(["accel-0", "gyro-0"]).window_slices == [
#         (0, 2),
#         (2, 4),
#     ]
#     assert merged.windows(["accel-1", "gyro-0"]).window_names == [
#         "accel-1",
#         "gyro-0",
#     ]
#     assert merged.windows(["accel-0", "accel-1", "gyro-0"]).window_slices == [
#         (0, 2), (2, 4), (4, 6)
#     ]

#     assert np.all(merged[2][0] == np.arange(40).reshape(5, 8)[2, 0:8])
#     assert np.all(merged[9][0] == np.arange(40).reshape(5, 8)[4, 0:8] * 2)
#     assert np.all(merged.windows(["accel-1", "gyro-0"])[9][0] == np.array([68, 70, 72, 74]))
#     assert merged[-1][1] == 8
#     assert merged[-2][1] != 7 
#     assert np.all(merged[:][1] == np.concatenate((np.arange(5), np.arange(5) *2)))

#     with pytest.raises(ValueError):
#         multimodal_array_1.concatenate(np.arange(40).reshape(5, 8))

# def test_dataset_join(multimodal_array_1: ArrayMultiModalDataset, multimodal_array_1_mag: ArrayMultiModalDataset):
#     joined = multimodal_array_1.join(multimodal_array_1_mag)
#     assert len(joined) == 5
#     assert joined.X.shape == (5, 14)
#     assert joined.y.shape == (5,)
#     assert joined.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1", "mag-0", "mag-1", "mag-2"]
#     assert joined.window_slices == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)]

#     assert joined.windows(["accel-0", "gyro-0"]).window_names == [
#         "accel-0",
#         "gyro-0",
#     ]
#     assert joined.windows(["accel-0", "gyro-0"]).window_slices == [
#         (0, 2),
#         (2, 4),
#     ]
#     assert joined.windows(["accel-1", "gyro-0"]).window_names == [
#         "accel-1",
#         "gyro-0",
#     ]
#     assert joined.windows(["accel-0", "accel-1", "mag-0"]).window_slices == [
#         (0, 2), (2, 4), (4, 6)
#     ]

#     assert np.all(joined[0][0] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 6, 9, 12, 15]))
#     assert np.all(joined[:][1] == np.arange(5))

#     with pytest.raises(ValueError):
#         multimodal_array_1.join(np.arange(40).reshape(5, 8))
