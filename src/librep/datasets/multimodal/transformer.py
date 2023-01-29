from typing import Optional, Tuple, List, Union
import uuid

import numpy as np
import pandas as pd

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

from .multimodal import (
    MultiModalDataset,
    ArrayMultiModalDataset,
    PandasMultiModalDataset,
)


# Calculate multi-class dice score from two numpy matrix using scikit-learn


def __label_selector(a, b):
    return a


def combine_multi_modal_datasets(
    d1: MultiModalDataset,
    d2: MultiModalDataset,
    collate_fn: callable = np.hstack,
    labels_combine: callable = __label_selector,
):
    new_X = collate_fn([d1[:][0], d2[:][0]])
    new_y = [labels_combine(y1, y2) for y1, y2 in zip(d1[:][1], d2[:][1])]

    last_slice_index = d1.window_slices[-1][1]
    window_slices = d1.window_slices + [
        (start + last_slice_index, end + last_slice_index)
        for start, end in d2.window_slices
    ]
    window_names = d1.window_names + d2.window_names

    return ArrayMultiModalDataset(new_X, new_y, window_slices, window_names)


class DatasetFitter:
    def __init__(
        self,
        transform: Transform,
        fit_on: Optional[Union[List[str], str]] = None,
        use_y: bool = False,
    ):
        self.the_transform = transform
        self.fit_on = fit_on
        self.use_y = use_y

    def fit(self, dataset: MultiModalDataset) -> "DatasetFitter":
        if self.fit_on is None:
            self.the_transform.fit(
                dataset[:][0], y=dataset[:][1] if self.use_y else None
            )
        else:
            dset = dataset.windows(self.fit_on)
            self.the_transform.fit(dset[:][0], y=dset[:][1] if self.use_y else None)
        return self

    def __call__(self, *args, **kwds):
        return self.fit(*args, **kwds)


class DatasetTransformer:
    def __init__(
        self,
        transform: Transform,
        transform_on: Optional[Union[List[str], str]] = None,
        new_suffix: str = "",
        join_window_names: bool = True,
        keep_other_windows: bool = False,
    ):
        self.the_transform = transform
        self.transform_on = transform_on
        self.new_suffix = new_suffix
        self.join_window_names = join_window_names
        self.keep_other_windows = keep_other_windows

    def transform(self, dataset: MultiModalDataset) -> ArrayMultiModalDataset:
        if self.transform_on is None:
            transformed_arr = self.the_transform.transform(dataset[:][0])
            name = ".".join(dataset.window_names) if self.join_window_names else ""
            name = f"{name}{self.new_suffix}"
            if not name:
                name = str(uuid.uuid4()[:8])

            return ArrayMultiModalDataset(
                X=transformed_arr,
                y=dataset[:][1],
                window_names=name,
                window_slices=[(0, transformed_arr.shape[1])],
            )
        else:
            remaining_windows = [
                w for w in dataset.window_names if w not in self.transform_on
            ]
            remaining_dataset = None
            if self.keep_other_windows and remaining_windows:
                if isinstance(dataset, PandasMultiModalDataset):
                    remaining_dataset = ArrayMultiModalDataset.from_pandas(
                        dataset.windows(remaining_windows)
                    )
                else:
                    remaining_dataset = dataset.windows(remaining_windows)

            arr = dataset.windows(self.transform_on)
            transformed_arr = self.the_transform.transform(arr[:][0])
            name = ".".join(arr.window_names) if self.join_window_names else ""
            name = f"{name}{self.new_suffix}"
            if not name:
                name = str(uuid.uuid4()[:8])

            arr = ArrayMultiModalDataset(
                X=transformed_arr,
                y=arr[:][1],
                window_names=[name],
                window_slices=[(0, transformed_arr.shape[1])],
            )
            if remaining_dataset is not None:
                return remaining_dataset.merge(arr)
            else:
                return arr

    def __call__(self, *args, **kwds):
        return self.transform(*args, **kwds)


class DatasetCombiner:
    def combine(self, *datasets):
        if len(datasets) < 2:
            raise ValueError("At least two datasets are required to combine.")
        dataset = datasets[0]
        if isinstance(dataset, PandasMultiModalDataset):
            dataset = ArrayMultiModalDataset.from_pandas(dataset)
        for d in datasets[1:]:
            if isinstance(d, PandasMultiModalDataset):
                d = ArrayMultiModalDataset.from_pandas(d)
            dataset = dataset.merge(d)
        return dataset

    def __call__(self, *args, **kwds):
        return self.combine(*args, **kwds)


class DatasetWindowedTransform:
    def __init__(
        self,
        transform: Transform,
        do_fit: bool = True,
        fit_on: List[str] = None,
        use_y: bool = False,
        transform_on: List[Union[str, List[str]]] = None,
        new_suffix: str = "",
        join_window_names: bool = True,
        combine: bool = True,
    ):
        self.the_transform = transform
        self.do_fit = do_fit
        self.fit_on = fit_on
        self.use_y = use_y
        self.transform_on = transform_on
        self.new_suffix = new_suffix
        self.join_window_names = join_window_names
        self.combine = combine

    def __call__(self, dataset: MultiModalDataset):
        if self.do_fit:
            DatasetFitter(
                transform=self.the_transform, fit_on=self.fit_on, use_y=self.use_y
            )(dataset)

        transform_on = (
            self.transform_on if self.transform_on is not None else dataset.window_names
        )
        datasets = []
        for window in transform_on:
            transformed_dset = DatasetTransformer(
                transform=self.the_transform,
                new_suffix=self.new_suffix,
                join_window_names=self.join_window_names,
                transform_on=window,
            )(dataset)
            datasets.append(transformed_dset)
        if self.combine:
            combiner = DatasetCombiner()
            return combiner(*datasets) if len(datasets) > 1 else datasets[0]
        else:
            return datasets


class DatasetSplitTransformCombine:
    def __init__(
        self,
        windows: List[Union[List[str], str]],
        transforms: List[Transform],
        new_suffixes: List[str] = None,
        join_window_names: bool = True,
        combine: bool = True,
    ):
        self.windows = windows
        self.transforms = transforms
        self.combine = combine
        self.new_suffixes = new_suffixes
        self.join_window_names = join_window_names

        if self.new_suffixes is None:
            self.new_suffixes = [""] * len(self.transforms)
        elif len(self.new_suffixes) != len(self.transforms):
            raise ValueError(
                "The number of new suffixes must match the number of transforms."
            )

    def transform(self, dataset: MultiModalDataset):
        datasets = []
        for window, suffix in zip(self.windows, self.new_suffixes):
            dset = dataset.windows(window)
            for transform in self.transforms:
                the_transform = DatasetTransformer(
                    transform,
                    new_suffix=suffix,
                    join_window_names=self.join_window_names,
                )
                dset = the_transform(dset)
            datasets.append(dset)

        if self.combine:
            combiner = DatasetCombiner()
            return combiner(*datasets) if len(datasets) > 1 else datasets[0]
        else:
            return datasets

    def __call__(self, *args, **kwds):
        return self.transform(*args, **kwds)


# class DatasetWindowedTransform:
#     def __init__(   self, transform: Transform, fit_on: str = "all", transform_on: str = "window",
#                     keep_remaining_windows: bool = True, select_windows: Union[List[str], str] = None,
#                     new_suffix: str = "", combiner: callable = combine_multi_modal_datasets):
#         self.the_transform = transform
#         assert fit_on in ["all", "window", None]
#         assert transform_on in ["all", "window", None]

#         if select_windows is not None and isinstance(select_windows, str):
#             self.select_windows = [select_windows]
#         else:
#             self.select_windows = select_windows
#         self.new_suffix = new_suffix

#     def fit(self, dataset: MultiModalDataset):
#         if self.select_windows is None:
#             new_set = dataset
#         else:
#             new_set = dataset.windows(self.select_windows)
#         self.the_transform.fit(new_set[:][0], new_set[:][1])
#         return self

#     def transform(self, dataset: MultiModalDataset):
#         if self.select_windows is None:
#             new_set = dataset
#         else:
#             new_set = dataset.windows(self.select_windows)

#         return ArrayMultiModalDataset(
#             self.the_transform.transform(new_set[:][0]),
#             y=new_set[:][1],
#             window_names=[f"{self.new_suffix}{n}"  for n in new_set.window_names],
#             window_slices=new_set.window_slices
#         )


class WindowedTransform:
    def __init__(
        self,
        transform: Transform,
        fit_on: str = "all",  # all, window or None
        transform_on: str = "window",  # all, window or None
        select_windows: List[str] = None,
        keep_remaining_windows: bool = True,
    ):
        self.the_transform = transform
        self.fit_on = fit_on
        self.transform_on = transform_on
        self.select_windows = select_windows
        self.keep_remaining_windows = keep_remaining_windows

        assert self.fit_on in ["all", "window", None]
        assert self.transform_on in ["all", "window", None]

        # if self.fit_on == "window":
        #     assert self.transform_on == "window" or self.transform_on is None


# class DatasetTransform:
#     def __init__(self, transform: Transform):
#         self.transform = transform

#     def fit(self, dataset: MultiModalDataset):
#         self.transform.fit(dataset[:][0], dataset[:][1])
#         return self

#     def transform(self, dataset: MultiModalDataset):
#         return ArrayMultiModalDataset(
#             dataset[:][0], y=dataset[:][1],
#             window_names=dataset.window_names,
#             window_slices=dataset.window_slices,
#         )

# class WindowedTransform2(Transform):
#     def __init__(self, transform: Transform, select_windows: List[str]):
#         self.the_transform = transform
#         self.select_windows = select_windows

#     def fit(self, dataset: MultiModalDataset):
#         new_set = dataset.windows(self.select_windows)
#         self.the_transform.fit(new_set[:][0], new_set[:][1])
#         return self

#     def transform(self, dataset: MultiModalDataset):
#         pass


class TransformMultiModalDataset:
    """Apply a list of transforms into the whole dataset, generating a new
    dataset.

    Parameters
    ----------
    transforms : List[Transform]
        List of transforms to be applyied to each sample, in order.

    Note: It supposes the number of windows will remain the same

    TODO: it not using fit. fit should be called over whole dataset.
    """

    def __init__(
        self,
        transforms: List[Transform],
        collate_fn: callable = np.hstack,
        new_window_name_prefix: str = "",
    ):
        self.transforms = transforms
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.collate_fn = collate_fn
        self.new_window_name_prefix = new_window_name_prefix

    def __transform_sample(
        self,
        transform: Transform,
        X: ArrayLike,
        y: ArrayLike,
        slices: List[Tuple[int, int]],
        do_fit: bool,
    ):
        if do_fit:
            return [
                transform.fit_transform(X[..., start:end], y) for start, end in slices
            ]
        else:
            return [transform.transform(X[..., start:end]) for start, end in slices]

    def split(self, dataset: MultiModalDataset, window_names: List[str]):
        new_X, new_slices, new_names = [], [], []
        i = 0
        for w in window_names:
            index = dataset.window_names.index(w)
            start, stop = dataset.window_slices[index]
            x = dataset[:][0][..., start:stop]
            new_X.append(x)
            new_slices.append((i, i + (stop - start)))
            new_names.append(w)
            i += stop - start
        return ArrayMultiModalDataset(
            self.collate_fn(new_X),
            y=dataset[:][1],
            window_slices=new_slices,
            window_names=new_names,
        )

    def __call__(self, dataset: MultiModalDataset):
        new_dataset = dataset
        for window_transform in self.transforms:
            if not isinstance(window_transform, WindowedTransform):
                window_transform = WindowedTransform(window_transform)

            select_windows = window_transform.select_windows or new_dataset.window_names
            selected_dataset = self.split(new_dataset, select_windows)
            X = selected_dataset[:][0]
            y = selected_dataset[:][1]

            # Combinations:
            # fit_on=None, transform_on=window *
            # fit_on=None, transform_on=all *
            # fit_on=window, transform_on=window *
            # fit_on=window, transform_on=all    (does not make sense)
            # fit_on=all, transform_on=window *
            # fit_on=all, transform_on=all *

            if window_transform.fit_on == "all":
                X, y = selected_dataset[:][0], selected_dataset[:][1]
                window_transform.the_transform.fit(X, y)
            elif window_transform.fit_on is None:
                pass

            if window_transform.transform_on == "window":
                # fit_on=None, transform_on=window *
                # fit_on=all, transform_on=window *
                # fit_on=window, transform_on=window *
                new_X = self.__transform_sample(
                    transform=window_transform.the_transform,
                    X=X,
                    y=y,
                    slices=selected_dataset.window_slices,
                    do_fit=window_transform.fit_on == "window",
                )
                new_y = y

                # Calculate new slices
                window_slices = []
                start = 0
                for x in new_X:
                    end = start + len(x[0])
                    window_slices.append((start, end))
                    start = end

                # Collate the windows into a single array
                new_X = self.collate_fn(new_X)
                new_y = np.array(new_y)
                new_slices = window_slices
                new_names = selected_dataset.window_names

                # Create a new dataset
                # new_dataset = ArrayMultiModalDataset(new_X, np.array(new_y),
                #                                      window_slices,
                #                                      new_dataset.window_names)

            else:
                # fit_on=all, transform_on=all *
                # fit_on=None, transform_on=all *
                new_X = window_transform.the_transform.transform(X=X)
                new_y = y
                new_slices = selected_dataset.window_slices
                new_names = selected_dataset.window_names

            selected_dataset = ArrayMultiModalDataset(
                new_X, new_y, new_slices, new_names
            )

            not_selected_windows = [
                w for w in new_dataset.window_names if w not in select_windows
            ]
            if not_selected_windows:
                if window_transform.keep_remaining_windows:
                    not_selected_dataset = self.split(new_dataset, not_selected_windows)
                    new_dataset = combine_multi_modal_datasets(
                        selected_dataset, not_selected_dataset
                    )
                else:
                    new_dataset = selected_dataset
            else:
                new_dataset = selected_dataset

        window_names = [
            f"{self.new_window_name_prefix}{name}" for name in new_dataset.window_names
        ]
        new_dataset._window_names = window_names
        return new_dataset
