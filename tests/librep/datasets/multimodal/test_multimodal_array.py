import pandas as pd
import numpy as np
from librep.datasets.multimodal.multimodal import (
    ArrayMultiModalDataset,
)
import pytest

@pytest.fixture
def multimodal_array_1():
    return ArrayMultiModalDataset(
        np.arange(40).reshape(5, 8),
        y=np.arange(5),
        window_names=["accel-0", "accel-1", "gyro-0", "gyro-1"],
        window_slices=[(0, 2), (2, 4), (4, 6), (6, 8)],
    )

@pytest.fixture
def multimodal_array_2():
    return ArrayMultiModalDataset(
        np.arange(40).reshape(5, 8) * 2,
        y=np.arange(5)*2,
        window_names=["accel-0", "accel-1", "gyro-0", "gyro-1"],
        window_slices=[(0, 2), (2, 4), (4, 6), (6, 8)],
    )

@pytest.fixture
def multimodal_array_1_mag():
    return ArrayMultiModalDataset(
        np.arange(30).reshape(5, 6) * 3,
        y=np.arange(5),
        window_names=["mag-0", "mag-1", "mag-2"],
        window_slices=[(0, 2), (2, 4), (4, 6)],
    )


def test_array_dataset_window(multimodal_array_1: ArrayMultiModalDataset):
    assert multimodal_array_1.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1"]
    assert multimodal_array_1.window_slices == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]
    assert multimodal_array_1.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1"]
    assert np.all(multimodal_array_1[0][0] == np.arange(40).reshape(5, 8)[0, 0:8])
    assert np.all(multimodal_array_1[2][0] == np.arange(40).reshape(5, 8)[2, 0:8])
    assert np.all(multimodal_array_1.windows("accel-1")[0][0] == np.arange(40).reshape(5, 8)[0, 2:4])
    assert np.all(multimodal_array_1.windows(["accel-0", "accel-1"])[0][0] == np.arange(40).reshape(5, 8)[0, 0:4])
    assert np.all(multimodal_array_1.windows(["gyro-0", "gyro-1"])[0][0] == np.arange(40).reshape(5, 8)[0, 4:8])
    assert np.all(multimodal_array_1.windows(["accel-1", "gyro-1"])[0][0] == np.array([2, 3, 6, 7]))
    assert np.all(multimodal_array_1.windows(["accel-0", "gyro-0"])[0][0] == np.array([0, 1, 4, 5]))
    assert multimodal_array_1.windows(["accel-0", "gyro-0"])[0][1] == 0
    assert multimodal_array_1.windows(["accel-0", "gyro-0"])[1][1] == 1
    assert multimodal_array_1.windows(["accel-0", "gyro-0"])[2][1] == 2
    assert multimodal_array_1.windows(["accel-0", "gyro-0"])[3][1] == 3
    assert multimodal_array_1.windows(["accel-0", "gyro-0"]).y[4] == 4

    assert multimodal_array_1.windows(["accel-0", "gyro-0"]).window_names == [
        "accel-0",
        "gyro-0",
    ]
    assert multimodal_array_1.windows(["accel-0", "gyro-0"]).window_slices == [
        (0, 2),
        (2, 4),
    ]
    assert multimodal_array_1.windows(["accel-1", "gyro-0"]).window_names == [
        "accel-1",
        "gyro-0",
    ]
    assert multimodal_array_1.windows(["accel-0", "accel-1", "gyro-0"]).window_slices == [
        (0, 2), (2, 4), (4, 6)
    ]

    with pytest.raises(ValueError):
        multimodal_array_1.windows(["accel-0", "accel-1", "gyro-0", "gyro-1", "mag-0"])
        multimodal_array_1.windows(["accel"])


def test_dataset_concatenate(multimodal_array_1: ArrayMultiModalDataset, multimodal_array_2: ArrayMultiModalDataset):
    merged = multimodal_array_1.concatenate(multimodal_array_2)
    assert len(merged) == 10

    assert merged.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1"]
    assert merged.window_slices == [(0, 2), (2, 4), (4, 6), (6, 8)]

    assert merged.windows(["accel-0", "gyro-0"]).window_names == [
        "accel-0",
        "gyro-0",
    ]
    assert merged.windows(["accel-0", "gyro-0"]).window_slices == [
        (0, 2),
        (2, 4),
    ]
    assert merged.windows(["accel-1", "gyro-0"]).window_names == [
        "accel-1",
        "gyro-0",
    ]
    assert merged.windows(["accel-0", "accel-1", "gyro-0"]).window_slices == [
        (0, 2), (2, 4), (4, 6)
    ]

    assert np.all(merged[2][0] == np.arange(40).reshape(5, 8)[2, 0:8])
    assert np.all(merged[9][0] == np.arange(40).reshape(5, 8)[4, 0:8] * 2)
    assert np.all(merged.windows(["accel-1", "gyro-0"])[9][0] == np.array([68, 70, 72, 74]))
    assert merged[-1][1] == 8
    assert merged[-2][1] != 7 
    assert np.all(merged[:][1] == np.concatenate((np.arange(5), np.arange(5) *2)))

    with pytest.raises(ValueError):
        multimodal_array_1.concatenate(np.arange(40).reshape(5, 8))

def test_dataset_join(multimodal_array_1: ArrayMultiModalDataset, multimodal_array_1_mag: ArrayMultiModalDataset):
    joined = multimodal_array_1.join(multimodal_array_1_mag)
    assert len(joined) == 5
    assert joined.X.shape == (5, 14)
    assert joined.y.shape == (5,)
    assert joined.window_names == ["accel-0", "accel-1", "gyro-0", "gyro-1", "mag-0", "mag-1", "mag-2"]
    assert joined.window_slices == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)]

    assert joined.windows(["accel-0", "gyro-0"]).window_names == [
        "accel-0",
        "gyro-0",
    ]
    assert joined.windows(["accel-0", "gyro-0"]).window_slices == [
        (0, 2),
        (2, 4),
    ]
    assert joined.windows(["accel-1", "gyro-0"]).window_names == [
        "accel-1",
        "gyro-0",
    ]
    assert joined.windows(["accel-0", "accel-1", "mag-0"]).window_slices == [
        (0, 2), (2, 4), (4, 6)
    ]

    assert np.all(joined[0][0] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 6, 9, 12, 15]))
    assert np.all(joined[:][1] == np.arange(5))

    with pytest.raises(ValueError):
        multimodal_array_1.join(np.arange(40).reshape(5, 8))
