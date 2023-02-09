from typing import Optional, List, Union
import uuid

from librep.base.estimator import Estimator
from librep.base.evaluators import Evaluators

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

from .multimodal import (
    MultiModalDataset,
    ArrayMultiModalDataset,
    PandasMultiModalDataset,
)

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
        return self.the_transform

    def __call__(self, *args, **kwds):
        return self.fit(*args, **kwds)

class DatasetPredicter:
    def __init__(
        self,
        estimator: Estimator,
        predict_on: Optional[Union[List[str], str]] = None,
        use_y: bool = False,
    ):
        self.the_estimator = estimator
        self.predict_on = predict_on
        self.use_y = use_y

    def predict(self, dataset: MultiModalDataset) -> ArrayLike:
        if self.predict_on is None:
            if self.use_y:
                return self.the_estimator.predict(dataset[:][0], y=dataset[:][1])
            else:
                return self.the_estimator.predict(dataset[:][0])

        else:
            dset = dataset.windows(self.predict_on)
            if self.use_y:
                return self.the_estimator.predict(dset[:][0], y=dset[:][1])
            else:
                return self.the_estimator.predict(dset[:][0])

    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)

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
                return remaining_dataset._merge(arr)
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
            dataset = dataset._merge(d)
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

class DatasetEvaluator:
    def __init__(self, evaluator: Evaluators):
        self.evaluator = evaluator

    def evaluate(self, *args, **kwds):
        return self.evaluator.evaluate(*args, **kwds)

    def __call__(self, *args, **kwds):
        return self.evaluate(*args, **kwds)

class DatasetX:
    def get(self, dataset: MultiModalDataset):
        return dataset[:][0]

    def __call__(self, *args, **kwds):
        return self.get(*args, **kwds)

class DatasetY:
    def get(self, dataset: MultiModalDataset):
        return dataset[:][1]

    def __call__(self, *args, **kwds):
        return self.get(*args, **kwds)     

class DatasetWindow:
    def __init__(self, window: Union[List[str], str]):
        if isinstance(window, str):
            self.window = [window]
        else:
            self.window = window
    
    def get(self, dataset: MultiModalDataset):
        return dataset.windows(self.window)

    def __call__(self, *args, **kwds):
        return self.get(*args, **kwds)

class Watcher:
    def __init__(self, func: callable) -> None:
        self.func = func

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)
