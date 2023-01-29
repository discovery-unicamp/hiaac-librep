import test_array
import pandas as pd
import numpy as np
from umap import UMAP
from librep.base.transform import Transform

from librep.config.type_definitions import ArrayLike
from librep.datasets.multimodal.transformer import (
    DatasetFitter,
    DatasetCombiner,
    DatasetTransformer,
    DatasetWindowedTransform,
)
from librep.datasets.multimodal.multimodal import (
    ArrayMultiModalDataset,
    PandasMultiModalDataset,
)


class SumTransform(Transform):
    def __init__(self):
        self.sum_val = 0

    def fit(self, X: ArrayLike, y: ArrayLike = None):
        self.sum_val = np.sum(X)

    def transform(self, X: ArrayLike):
        return X + self.sum_val

    def __str__(self) -> str:
        return f"SumTransform(sum_val={self.sum_val})"

    def __repr__(self) -> str:
        return str(self)


def multimodal_dataframe_1():
    df = pd.DataFrame(
        np.arange(80).reshape(10, 8),
        columns=[
            "accel-0",
            "accel-1",
            "accel-2",
            "accel-3",
            "gyro-0",
            "gyro-1",
            "gyro-2",
            "gyro-3",
        ],
    )
    df["label"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    return PandasMultiModalDataset(
        df, feature_prefixes=["accel", "gyro"], label_columns=["label"]
    )


def multimodal_dataframe_2():
    df = pd.DataFrame(
        np.arange(40).reshape(5, 8) * -2,
        columns=[
            "accel-0",
            "accel-1",
            "accel-2",
            "accel-3",
            "gyro-0",
            "gyro-1",
            "gyro-2",
            "gyro-3",
        ],
    )
    df["label"] = ["f", "g", "h", "i", "j"]
    return PandasMultiModalDataset(
        df, feature_prefixes=["accel", "gyro"], label_columns=["label"]
    )


dset_1 = multimodal_dataframe_1()
print(dset_1)

print("Window: accel")
print(dset_1)
print(dset_1.window_names)
print(dset_1.window_slices)

# dset_2 = multimodal_dataframe_2()
my_transform = UMAP(n_neighbors=5, n_components=2, transform_seed=42, random_state=42)

print("--------- fit")
fit = DatasetFitter(my_transform, fit_on=["accel"])
fit(dset_1)
print("----- Transform 1")
transform_accel = DatasetTransformer(
    my_transform,
    transform_on=["accel"],
    new_suffix="_reduced",
    keep_other_windows=False,
)
acc_dset = transform_accel(dset_1)
print(acc_dset)
print(acc_dset.window_names)
print(acc_dset.window_slices)

print("----- Transform 2")
transform_gyro = DatasetTransformer(
    my_transform, transform_on=["gyro"], new_suffix="_reduced", keep_other_windows=False
)
gyro_dset = transform_gyro(dset_1)
print(gyro_dset)
print(gyro_dset.window_names)
print(gyro_dset.window_slices)
print("----- Combine")
combine = DatasetCombiner()
dset_combined = combine(acc_dset, gyro_dset)
print(dset_combined)
print(dset_combined.window_names)
print(dset_combined.window_slices)

print(dset_combined)
print(dset_combined.window_names)

print("-----------------------------------")

dset_1 = multimodal_dataframe_1()
my_transform = SumTransform()
transformer = DatasetWindowedTransform(
    my_transform,
    do_fit=True,
    fit_on=["accel"],
    use_y=False,
    transform_on=None,
    new_suffix="_sum",
    join_window_names=True,
)
dset_transformed = transformer(dset_1)
print(dset_transformed)
print(dset_transformed.window_names)
print(dset_transformed.window_slices)


# print("--------- All")

# transform_all = DatasetTransformer(my_transform, transform_on=None)
# dset_combined = transform_all(dset_1)
# print(dset_combined)
# print(dset_combined.window_names)


# print(dset_1[0][0])
# print(dset_combined[0][0])
# print(dset_combined[0][1])
# print()


# from librep.datasets.multimodal.multimodal import ArrayMultiModalDataset, PandasMultiModalDataset
# from librep.datasets.multimodal.transformer import TransformMultiModalDataset, WindowedTransform
# from librep.transforms.fft import FFT

# from librep.datasets.multimodal import PandasMultiModalDataset

# dataframe = pd.DataFrame(
#     np.arange(20).reshape(4, 5),
#     columns=["accel-0", "accel-1", "gyro-0", "gyro-1", "label"]
# )

# dataset = ArrayMultiModalDataset(
#     np.arange(20).reshape(5, 4), y=np.arange(5), window_names=["accel-0", "accel-1", "gyro-0", "gyro-1"], window_slices=[(0, 1), (1, 2), (2, 3), (3, 4)]
# )
# print(dataset, dataset[0])

# x = dataset.windows(["accel-0", "gyro-0"])
# print(x, x[0])

# fft = FFT(centered=True)
# transform_fit = WindowedTransform(fft, fit_on="window", transform_on=None, select_windows=["accel-0", "gyro-0"])
# transform_2 = WindowedTransform(fft, fit_on=None, transform_on="window", select_windows=["accel-0", "gyro-0"])
# transformer = TransformMultiModalDataset(transforms=[transform_fit, transform_2])
# y = transformer(x)
# print(y, y[0])


print("------------\n")
