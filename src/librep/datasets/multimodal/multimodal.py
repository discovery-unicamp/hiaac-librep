from typing import Iterable, Tuple, Optional, Union, List

import numpy as np
import pandas as pd

from librep.base.data import Dataset
from librep.datasets.common import PandasDataset
from librep.config.type_definitions import ArrayLike


class MultiModalDataset(Dataset):
    @property
    def window_slices(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @property
    def window_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def num_windows(self) -> int:
        raise NotImplementedError

    def windows(self, names: Union[str, List[str]]) -> "MultiModalDataset":
        raise NotImplementedError

    def _merge(self, other: "MultiModalDataset") -> "MultiModalDataset":
        raise NotImplementedError

    def _concatenate(self, other: "MultiModalDataset") -> "MultiModalDataset":
        raise NotImplementedError


class ArrayMultiModalDataset(MultiModalDataset):
    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        window_slices: List[Tuple[int, int]],
        window_names: List[str] = None,
        collate_fn: callable = np.hstack,
    ):
        self.X = X
        self.y = y
        self._window_slices = window_slices
        self._window_names = window_names or []
        self.collate_fn = collate_fn

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def window_slices(self) -> List[Tuple[int, int]]:
        return self._window_slices

    @property
    def window_names(self) -> List[str]:
        return self._window_names

    @property
    def num_windows(self) -> int:
        return len(self._window_slices)

    def windows(self, names: Union[str, List[str]]) -> "ArrayMultiModalDataset":
        if isinstance(names, str):
            names = [names]

        new_X = []
        window_slices = []
        window_names = []
        last_slice = (0, 0)
        for name in names:
            if name not in self.window_names:
                raise ValueError(f"Window name {name} not found")
            the_slice = self.window_slices[self.window_names.index(name)]
            window_slices.append(
                (last_slice[1], the_slice[1] - the_slice[0] + last_slice[1])
            )
            last_slice = window_slices[-1]
            window_names.append(name)
            new_X.append(self.X[:, the_slice[0] : the_slice[1]])

        new_X = self.collate_fn(new_X)
        return ArrayMultiModalDataset(
            new_X, self.y, window_slices=window_slices, window_names=window_names
        )

    def _merge(self, other: "ArrayMultiModalDataset") -> "ArrayMultiModalDataset":
        if not isinstance(other, ArrayMultiModalDataset):
            raise ValueError("Can only join with ArrayMultiModalDataset")

        selector = lambda a, b: a

        X = self.collate_fn([self.X, other.X])
        y = np.array([selector(a, b) for a, b in zip(self.y, other.y)])
        last_slice_index = self.window_slices[-1][1]
        window_slices = self.window_slices + [
            (s[0] + last_slice_index, s[1] + last_slice_index)
            for s in other.window_slices
        ]
        window_names = self.window_names + other.window_names

        return ArrayMultiModalDataset(
            X, y, window_slices=window_slices, window_names=window_names
        )

    def _concatenate(self, other: "ArrayMultiModalDataset") -> "ArrayMultiModalDataset":
        if not isinstance(other, ArrayMultiModalDataset):
            raise ValueError("Can only join with ArrayMultiModalDataset")
        if len(self.window_slices) != len(other.window_slices):
            raise ValueError("Both datasets must have the same number of windows")
        for s1, s2 in zip(self.window_slices, other.window_slices):
            if s1[0] != s2[0] or s1[1] != s2[1]:
                raise ValueError("Both datasets must have the same window slices")
        for n1, n2 in zip(self.window_names, other.window_names):
            if n1 != n2:
                raise ValueError("Both datasets must have the same window names")

        X = np.concatenate([self.X, other.X])
        y = np.concatenate([self.y, other.y])
        return ArrayMultiModalDataset(
            X, y, window_slices=self.window_slices, window_names=self.window_names
        )

    @staticmethod
    def from_pandas(other: "PandasMultiModalDataset") -> "ArrayMultiModalDataset":
        new_start = 0
        slices = []
        for start, end in other.window_slices:
            slices.append((new_start, new_start + end - start))
            new_start += end - start
        return ArrayMultiModalDataset(
            X=other[:][0],
            y=other[:][1],
            window_slices=slices,
            window_names=other.window_names,
        )

    def __str__(self):
        return f"{str(self.X)}\n{repr(self)}"

    def __repr__(self) -> str:
        return f"ArrayMultiModalDataset: samples={len(self.X)}, shape={self.X.shape}, no. window={self.num_windows}"


class PandasMultiModalDataset(PandasDataset, MultiModalDataset):
    """Dataset implementation for multi modal PandasDataset.
    It assumes that each sample is composed is a feature vector where
    parts of this vector comes from different natures.
    For instance, a sample with 900 features where features:
    - 0-299: correspond to acelerometer x
    - 300-599: correspond to accelerometer y
    - 600-899: correspond to accelerometer z

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_prefixes: Optional[Union[str, List[str]]] = None,
        label_columns: Union[str, List[str]] = "activity code",
        as_array: bool = True,
    ):
        """The MultiModalHAR dataset, derived from Dataset.
        The __getitem__ returns 2-element tuple where:
        - The first element is the sample (from the indexed-row of the
        dataframe with the selected features, as features); and
        - The seconds element is the label (from the indexed-row of the
        dataframe with the selected label_columns, as labels) .

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe with KuHar samples.
        feature_prefixes : Optional[Union[str, List[str]]]
            Which features from features must be selected. Features will be
            selected based on these prefixes. If None, select all features.
        label_columns : Union[str, List[str]]
            The columns(s) that represents the label. If the value is an `str`,
            a scalar will be returned, else, a list will be returned.
        as_array : bool
            If true, return a `np.ndarray`, else return a `pd.Series`, for each
            sample.

        Examples
        ----------
        >>> train_csv = pd.read_csv(my_filepath)
        >>> # This will select the accelerometer (x, y, and z) from HAR dataset
        >>> train_dataset = MultiModalHARDataset(feature_prefixes=["accel-x", "accel-y", "accel-z"], label_columns="activity code")
        >>> len(train_dataset)
        10
        >>> train_dataset[0]
        (np.ndarray(0.5, 0.6, 0.7), 0)

        """

        self.feature_prefixes = feature_prefixes

        if feature_prefixes is None:
            to_select_features = list(set(dataframe.columns) - set(label_columns))
            self.feature_windows = [
                {
                    "prefix": "all",
                    "start": 0,
                    "end": len(to_select_features),
                    "features": to_select_features,
                }
            ]
        else:
            if isinstance(feature_prefixes, str):
                feature_prefixes = [feature_prefixes]

            start = 0
            self.feature_windows = []

            for prefix in feature_prefixes:
                features = [col for col in dataframe.columns if col.startswith(prefix)]
                end = start + len(features)
                self.feature_windows.append(
                    {"prefix": prefix, "start": start, "end": end, "features": features}
                )
                start = end

            to_select_features = [
                col
                for prefix in feature_prefixes
                for col in dataframe.columns
                if col.startswith(prefix)
            ]

        super().__init__(
            dataframe,
            features_columns=to_select_features,
            label_columns=label_columns if isinstance(label_columns, list) else [label_columns],
            as_array=as_array,
        )

    @property
    def window_slices(self) -> List[Tuple[int, int]]:
        return [(window["start"], window["end"]) for window in self.feature_windows]

    @property
    def window_names(self) -> List[str]:
        return [window["prefix"] for window in self.feature_windows]

    @property
    def num_windows(self) -> int:
        return len(self.window_slices)

    def windows(self, names: Union[str, List[str]]) -> "PandasMultiModalDataset":
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in self.window_names:
                raise ValueError(f"Window '{name}' not found.")

        return PandasMultiModalDataset(
            dataframe=self.data,
            feature_prefixes=names,
            label_columns=self.label_columns,
            as_array=self.as_array,
        )

    def _merge(self, other: "PandasMultiModalDataset") -> "PandasMultiModalDataset":
        if not isinstance(other, PandasMultiModalDataset):
            raise ValueError("Can only join with PandasMultiModalDataset")

        # Check if self.data and other.data has any column name that is equal, excluding the self.label columns..
        label_columns = set([self.label_columns] if isinstance(self.label_columns, str) else self.label_columns)
        other_label_columns = set([other.label_columns] if isinstance(other.label_columns, str) else other.label_columns)
        if (
            len(
                (set(self.data.columns) - label_columns)
                & (set(other.data.columns) - other_label_columns)
            )
            > 0
        ):
            raise ValueError(
                "Can only join with PandasMultiModalDataset with different column names"
            )

        data = pd.concat([self.data, other.data], axis=1).reset_index(drop=True)
        feature_windows = self.feature_windows + [
            {
                "prefix": window["prefix"],
                "start": window["start"] + self.data.shape[-1],
                "end": window["end"] + self.data.shape[-1],
                "features": window["features"],
            }
            for window in other.feature_windows
        ]

        dset = PandasMultiModalDataset(
            data,
            feature_prefixes=self.feature_prefixes+other.feature_prefixes,
            label_columns=self.label_columns,
            as_array=self.as_array,
        )
        dset.data = dset.data.loc[:, ~dset.data.columns.duplicated()]
        dset.feature_windows = feature_windows
        dset.feature_columns = self.feature_columns + other.feature_columns
        return dset

    def _concatenate(self, other: "PandasDataset") -> "PandasDataset":
        if not isinstance(other, PandasMultiModalDataset):
            raise ValueError("Can only concatenate with PandasMultiModalDataset")

        if self.label_columns != other.label_columns:
            raise ValueError(
                "Can only concatenate with PandasMultiModalDataset with same label columns"
            )

        df = pd.concat([self.data, other.data], axis=0).reset_index(drop=True)
        return PandasMultiModalDataset(
            df,
            feature_prefixes=self.feature_prefixes,
            label_columns=self.label_columns,
            as_array=self.as_array,
        )

    def get_data(self) -> pd.DataFrame:
        return self.data[self.feature_columns]

    @staticmethod
    def from_array(
        other: ArrayMultiModalDataset, label_column_name="label", as_array: bool = True
    ) -> "PandasMultiModalDataset":
        """Create a `PandasMultiModalDataset` from a `ArrayMultiModalDataset`."""
        if not isinstance(other, ArrayMultiModalDataset):
            raise ValueError("dataset must be an ArrayMultiModalDataset")

        X, y = other.X.copy(), other.y.copy()
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]

        if y.shape[-1] == 1:
            label_cols = [label_column_name]
        else:
            label_cols = [f"{label_column_name}-{i}" for i in range(y.shape[-1])]

        columns = [f"Unnamed: {i}" for i in range(X.shape[-1])] + label_cols
        for wname, (start, end) in zip(other.window_names, other.window_slices):
            columns[start:end] = [f"{wname}-{i-start}" for i in range(start, end)]

        data = pd.DataFrame(np.hstack([X, y]), columns=columns)
        return PandasMultiModalDataset(
            data,
            feature_prefixes=other.window_names,
            label_columns=label_cols if len(label_cols) > 1 else label_cols[0],
            as_array=as_array,
        )

    def __str__(self) -> str:
        return f"{str(self.get_data())}\n{repr(self)}"

    def __repr__(self) -> str:
        return f"PandasMultiModalDataset: samples={len(self.data)}, features={len(self.feature_columns)}, no. window={self.num_windows}, label_columns='{self.label_columns}'"


def concat(datasets: List[MultiModalDataset], axis: int = 0) -> MultiModalDataset:
    """Concatenate a list of `MultiModalDataset`."""
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")
    if not isinstance(datasets, list) and not isinstance(datasets, tuple):
        raise ValueError("datasets must be a list")
    if len(datasets) < 1:
        raise ValueError("datasets must contain at least two datasets")

    dataset = datasets[0]
    for other in datasets[1:]:
        if not isinstance(other, type(dataset)):
            raise ValueError(
                "Can only concatenate datasets of the same type. Explicitly cast to the same type first."
            )
        if axis == 0:
            dataset = dataset._concatenate(other)
        else:
            dataset = dataset._merge(other)
    return dataset
