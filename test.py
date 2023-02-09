import pandas as pd
import numpy as np
from librep.datasets.multimodal.multimodal import (
    ArrayMultiModalDataset,
    PandasMultiModalDataset,
    concat
)


def multimodal_array_1():
    return ArrayMultiModalDataset(
        np.arange(40).reshape(5, 8),
        y=np.arange(5),
        window_names=["accel-x", "accel-y", "gyro-x", "gyro-y"],
        window_slices=[(0, 2), (2, 4), (4, 6), (6, 8)],
    )

def multimodal_array_1_mag():
    return ArrayMultiModalDataset(
        np.arange(30).reshape(5, 6) * 3,
        y=np.arange(5),
        window_names=["mag-x", "mag-y", "mag-z"],
        window_slices=[(0, 2), (2, 4), (4, 6)],
    )


x = multimodal_array_1()
y = multimodal_array_1_mag()
x = PandasMultiModalDataset.from_array(x)
y = PandasMultiModalDataset.from_array(y, label_column_name="loss")
z = concat((x,y), axis=1)
a = ArrayMultiModalDataset.from_pandas(z)
print(z[0])
print(a[0])