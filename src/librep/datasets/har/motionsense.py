import glob
import os
import random
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Generator

import numpy as np
import pandas as pd
import tqdm

from librep.config.type_definitions import PathLike
#from librep.utils.file_ops import download_unzip_check
from librep.datasets.har.generator import HARDatasetGenerator, DatasetSplitError

from scipy import signal


from librep.utils.file_ops import WgetDownload, ZipExtractor, DownloaderExtractor

##### Raw data Handlers and time series generator


class RawMotionSense:
    """This class handles raw files from MotionSense Dataset.

    Parameters
    ----------
    dataset_dir : PathLike
        Path to the root of motion sense dataset.
    download : bool
        If the dataset must be downloaded before (the default is False).

    Attributes
    ----------
    metadata_df : pd.DataFrame
        Dataframe relating user and activities to the their respective files.
    activity_names : Dict[int, str]
        Dictionary relating activity codes to activity names
    activity_codes : Dict[str, int]
        Dictionary relating activity names to activity codes
    dataset_dir: Path
        Path to the root of motion sense dataset.
    """

    # Version 1 MotionSense (Device Motion Data)
    dataset_url = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"
    data_path: str = "A_DeviceMotion_data"

    # Activity names and codes
    activity_names = {0: "dws", 1: "ups", 2: "sit", 3: "std", 4: "wlk", 5: "jog"}

    activity_codes = {v: k for k, v in activity_names.items()}

    def __init__(self, dataset_dir: PathLike, download: bool = False):
        self.dataset_dir = Path(dataset_dir)
        if download:
            self.__download_and_extract()
        # Metadata relates activities and users to their respective files.
        self.metadata_df: pd.DataFrame = self.__read_metadata()

    def __download_and_extract(self):
        """Download and extract the MotionSense dataset (A only).

        """
        # Create directories
        self.dataset_dir.mkdir(exist_ok=True, parents=True)

        # Download and extract
        downloader = DownloaderExtractor(
            downloader_cls=WgetDownload,
            extractor_cls=ZipExtractor,
            checker_cls=None
        )
        downloader.download_extract_check(
            url=self.dataset_url,
            destination_download_file="motionsense.zip",
            checksum=None,
            extract_folder=self.dataset_dir,
            remove_downloads=True
        )

        source_path = self.dataset_dir / self.data_path
        for file in source_path.glob("*"):
            file.rename(self.dataset_dir / file.name)
        source_path.rmdir()
        
        mac_os_X_path = self.dataset_dir / "__MACOSX"
        shutil.rmtree(str(mac_os_X_path))

        print(f"Data downloaded and extracted to {self.dataset_dir}")



    def __read_metadata(self) -> pd.DataFrame:
        """Iterate over dataset files and create a metadata dataframe.
            The metadata relates user and activities to their respective files.

        Returns
        -------
        pd.DataFrame
            Metadata relating users and activities to their respective CSV files.

        """
        # Let's list all CSV files in the directory
        files = self.dataset_dir.rglob("*.csv")

        # And create a relation of each user, activity and CSV file
        users_relation = []
        for f in files:
            # Pick activity name (folder name, e.g.: dws_1)
            activity_name = f.parents[0].name
            # Split activity name and trial code(e.g.: ['dws', 1])
            act_name, trial_code = activity_name.split("_")
            if trial_code == "checkpoints":
                continue
            else:
                trial_code = int(trial_code)
            # Get the activity number from the activity's code
            act_no = self.activity_codes[act_name]
            # Get user code
            user = f.stem.split("_")[1]
            user = int(user)
            # Generate a tuple of infotmation
            users_relation.append(
                {
                    "activity code": act_no,
                    "user": user,
                    "trial_code": trial_code,
                    "file": str(f),
                }
            )

        # Create a dataframe with all meta information
        column_dtypes = [
            ("activity code", np.int),
            ("user", np.int),
            ("trial_code", np.int),
            ("file", str),
        ]
        metadata_df = pd.DataFrame(
            users_relation, columns=[d[0] for d in column_dtypes]
        )
        for name, t in column_dtypes:
            metadata_df[name] = metadata_df[name].astype(t)
        return metadata_df

    def read_information(
        self, file: PathLike, activity_code: int, user: int, trail: Optional[int] = None
    ) -> pd.DataFrame:
        """Read the information of an user/activity, based on the metadata.

        Parameters
        ----------
        file : PathLike
            Path to the CSV with from user and activity.
            Can be retrieved from the metadata dataframe.
        activity_code : int
            The activity code.
        user : int
            The ID of the user.
        trail : Optional[int]
            The trial number.

        Returns
        -------
        pd.DataFrame
            Dataframe with the information of the user.

        """

        file = Path(file)
        # Default feature names from this dataset
        feature_dtypes = {
            "attitude.roll": np.float,
            "attitude.pitch": np.float,
            "attitude.yaw": np.float,
            "gravity.x": np.float,
            "gravity.y": np.float,
            "gravity.z": np.float,
            "rotationRate.x": np.float,
            "rotationRate.y": np.float,
            "rotationRate.z": np.float,
            "userAcceleration.x": np.float,
            "userAcceleration.y": np.float,
            "userAcceleration.z": np.float,
        }

        with file.open("r") as f:
            csv_matrix = pd.read_csv(
                f, names=list(feature_dtypes.keys()), dtype=feature_dtypes, skiprows=1
            )
            # csv_matrix["User"] = info["User"]
            # csv_matrix["Action Code"] = info["Action Code"]
            # csv_matrix["Sequence"] = info["Sequence"]

            # Reordering to same format as all MotionSense datasets
            csv_matrix = csv_matrix[
                [
                    "attitude.roll",
                    "attitude.pitch",
                    "attitude.yaw",
                    "gravity.x",
                    "gravity.y",
                    "gravity.z",
                    "rotationRate.x",
                    "rotationRate.y",
                    "rotationRate.z",
                    "userAcceleration.x",
                    "userAcceleration.y",
                    "userAcceleration.z",
                ]
            ]
            csv_matrix["activity code"] = activity_code
            csv_matrix["length"] = 1
            csv_matrix["trial_code"] = trail
            csv_matrix["index"] = range(len(csv_matrix))
            csv_matrix["user"] = user
            return csv_matrix

    @property
    def users(self) -> List[int]:
        return np.sort(self.metadata_df["user"].unique()).tolist()

    @property
    def activities(self) -> List[int]:
        return np.sort(self.metadata_df["activity code"].unique()).tolist()

    def __str__(self):
        return f"MotionSense Dataset at: '{self.dataset_dir}' ({len(self.metadata_df)} files)"

    def __repr__(self):
        return f"MotionSense Dataset at: '{self.dataset_dir}'"


class RawMotionSenseIterator:
    """Iterate over RawMotionSense files.
        Return a dataframe with samples from an user/activity.

    Parameters
    ----------
    motionsense : RawMotionSense
        RawMotionSense data files object.
    users_to_select : List[str]
        Users to select. If None, iterate over all users.
    activities_to_select : List[str]
        Activities to select. If None, iterate over all activities.
    shuffle : bool
        If must iterate randomly.

    """

    def __init__(
        self,
        motionsense: RawMotionSense,
        users_to_select: Optional[List[int]] = None,
        activities_to_select: Optional[List[str]] = None,
        shuffle: bool = False,
    ):
        self.motionsense = motionsense
        self.users_to_select = (
            users_to_select if users_to_select is not None else motionsense.users
        )
        self.activities_to_select = (
            activities_to_select
            if activities_to_select is not None
            else motionsense.activities
        )
        self.shuffle = shuffle
        self.it = None

    def __get_data_iterator(self) -> Generator[pd.DataFrame, None, None]:
        """Get an iterator to iterate over selected dataframes.

        Returns
        -------
        pd.DataFrame
            A dataframe for a user/activity.

        """
        selecteds = self.motionsense.metadata_df[
            (self.motionsense.metadata_df["user"].isin(self.users_to_select))
            & (
                self.motionsense.metadata_df["activity code"].isin(
                    self.activities_to_select
                )
            )
        ]

        # Shuffle data
        if self.shuffle:
            selecteds = selecteds.sample(frac=1)

        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            act_code = row["activity code"]
            user = row["user"]
            trial = row["trial_code"]
            file = row["file"]
            data = self.motionsense.read_information(
                file=file, activity_code=act_code, user=user, trail=trial
            )
            yield data

    def __str__(self) -> str:
        return f"MotionSense Iterator: users={len(self.users_to_select)}, activities={len(self.activities_to_select)}"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        self.it = self.__get_data_iterator()
        return self.it

    def __next__(self):
        return next(self.it)


class MotionSenseDatasetGenerator(HARDatasetGenerator):
    """Generate a custom MotionSense dataset from Raw MotionSense data.

    Parameters
    ----------
    motionsense_iterator : RawMotionSenseIterator
        The iterator object to iterate over users/activities dataframes.
    time_window : int
        Number of samples that compose a window.
        If None, a sample will be a single instant.
    window_overlap : int
        Number of samples to overlap over windows.
    add_gravity: bool
        Parameter to sum the gravity.
        If True will sum the gravity in the data
    add_filter: bool
        Parameter to apply the highpass filter over data.
        If True will apply a Butteworth filter.
    fs: int
        Number to change the frequency data.
    resampler: bool
        Parameter to resample the signal.
        If true it resample the signal.
    change_acc_measure: bool
        Parameter to change accelerometer measure.
        If true it times acceerometer signal by 9.81.
    """

    def __init__(
        self,
        motionsense_iterator: RawMotionSenseIterator,
        time_window: Optional[int] = None,
        window_overlap: Optional[int] = None,
        fs: int = None,
        resampler: bool = False,
        change_acc_measure: bool = False,
        add_gravity: bool = True,
        add_filter: bool = False,
    ):
        self.motionsense_iterator = motionsense_iterator
        self.time_window = time_window
        self.window_overlap = window_overlap
        self.fs = fs
        self.resampler = resampler
        self.change_acc_measure = change_acc_measure
        self.add_gravity = add_gravity
        self.add_filter = add_filter

        if window_overlap is not None:
            assert (
                time_window is not None
            ), "Time window must be set when overlap is set"

        if add_filter is True:
            assert (
                add_gravity is True
            ), "Add filter must be true when add gravity is true"

        if resampler is True:
            assert (
                fs is not None
            ), "Resampler must be set when fs is set"

    def __create_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a time series with defined window size and overlap.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be splitted to windows. Windows consist in consecutive
            samples as features.

        Returns
        -------
        pd.DataFrame
            Dataframe with time windows.

        """

        values = []
        column_names = []

        selected_features = [
            "attitude.roll",
            "attitude.pitch",
            "attitude.yaw",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
        ]

        # Parameter to sum the gravity 
        if self.add_gravity is True:
            data["userAcceleration.x"] = data["userAcceleration.x"] + data["gravity.x"]
            data["userAcceleration.y"] = data["userAcceleration.y"] + data["gravity.y"]
            data["userAcceleration.z"] = data["userAcceleration.z"] + data["gravity.z"] 

        # Change the accelerometer unit of measurement from g to m/s²      
        if self.change_acc_measure is True:
            data["userAcceleration.x"] = (data["userAcceleration.x"])*9.81
            data["userAcceleration.y"] = (data["userAcceleration.y"])*9.81
            data["userAcceleration.z"] = (data["userAcceleration.z"])*9.81

        if self.add_filter is True:
            # fs=50 because it's the original sample rate from dataset
            h = signal.butter(3, .3, 'hp', fs=50, output='sos')

            for axi in selected_features[-3:]:
                sig = data[axi]
                sample_filtered = signal.sosfiltfilt(h, sig)

                data[axi] = sample_filtered
                
        # Resampling the signal from 50Hz to fs

        time = data.shape[0] // 50
        new_data = {column: [] for column in selected_features}
        if self.resampler is True:
            for column in selected_features:
                new_data[column] = signal.resample(data[column], self.fs*time)
            new_data = pd.DataFrame(data=new_data)

        else:
            new_data = data[selected_features].copy()
            
        tam = new_data.shape[0]
        new_data['activity code'] = data['activity code'].iloc[:tam]
        new_data['length'] = data['length'].iloc[:tam]
        new_data['trial_code'] = data['trial_code'].iloc[:tam]
        new_data['index'] = data['index'].iloc[:tam]
        new_data['user'] = data['user'].iloc[:tam]
        
        data = new_data.copy()
        
        for i in range(0, data.shape[0], self.time_window - self.window_overlap):
            window_df = data[i : i + self.time_window]
            # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window
            window_values = window_df[selected_features].unstack().to_numpy()
            # acc_time = window_df["accel-start-time"].iloc[0], window_df["accel-end-time"].iloc[-1]
            # gyro_time = window_df["gyro-start-time"].iloc[0], window_df["gyro-end-time"].iloc[-1]
            act_class = window_df["activity code"].iloc[0]
            length = self.time_window
            trial_code = window_df["trial_code"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate(
                (
                    window_values,
                    [
                        act_class,
                        length,
                        trial_code,
                        start_idx,
                        act_user,
                    ],
                )
            )
            values.append(temp)

        # Name the cows
        column_names = [
            f"{feat}-{i}" for feat in selected_features for i in range(self.time_window)
        ]
        column_names += [c for c in data.columns if c not in selected_features]
        df = pd.DataFrame(values, columns=column_names)
        # Drop non values (remove last rows that no. samples does not fit window size)
        df = df.dropna()

        # Hack to maintain types
        for c in ["activity code", "length", "trial_code", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def get_full_df(self, use_tqdm: bool = True) -> pd.DataFrame:
        """Concatenate dataframe from windows.

        Parameters
        ----------
        use_tqdm : bool
            If must use tqdm as iterator (the default is True).

        Returns
        -------
        pd.DataFrame
            A single dataframe, with all dataframe of windows, concatenated.

        """

        it = iter(self.motionsense_iterator)
        if use_tqdm:
            it = tqdm.tqdm(
                it,
                desc="Generating full df over MotionSense View",
                position=0,
                leave=True,
            )

        if self.time_window is None:
            return pd.concat(it)
        else:
            return pd.concat(self.__create_time_series(d) for d in it)

    def check_if_unique_per_df(
        self,
        dataset_to_check: pd.DataFrame,
        datasets_list: List[pd.DataFrame],
        column: str = "user",
    ) -> bool:
        def get_uniques(df):
            return list(df[column].unique())

        column_in_ds_to_check = get_uniques(dataset_to_check)
        for ds in datasets_list:
            user_in_ds = get_uniques(ds)
            for i in column_in_ds_to_check:
                if i in user_in_ds:
                    return False
        return True

    def train_test_split(
        self,
        df: pd.DataFrame,
        users: List[int],
        activities: List[int],
        train_size: float,
        validation_size: float,
        test_size: float,
        retries: int = 10,
        ensure_distinct_users_per_dataset: bool = True,
        seed: int = None,
    ):
        n_users = len(users)

        for i in range(retries):
            # [start ---> train_size)
            random.shuffle(users)
            train_users = users[0:int(n_users * train_size)]
            # [train_size --> train_size+validation_size)
            validation_users = users[
                int(n_users * train_size):
                int(n_users * (train_size + validation_size))
            ]
            # [train_size+validation_size --> end]
            test_users = users[int(n_users * (train_size + validation_size)):]
            # iterate over user's lists, filter df for users in the respective list
            all_sets = [
                df[df["user"].isin(u)]
                for u in [train_users, validation_users, test_users]
            ]

            if not ensure_distinct_users_per_dataset:
                return all_sets

            # We must guarantee that all sets contains at least 1 sample from each activities listed
            oks = [set(s["activity code"]) == set(activities) for s in all_sets]
            if all(oks):
                # If all sets contains at least 1 sample for each activity, return train, val, test sets!
                return all_sets

        raise DatasetSplitError(
            "Does not found a 3 sets that contain the respective activities!"
        )

    def balance_dataset_to_minimum(
        self, dataframe: pd.DataFrame, column: str = "activity code"
    ) -> pd.DataFrame:
        df_list = []
        histogram = dataframe.groupby(dataframe[column], as_index=False).size()
        for c in histogram[column]:
            temp = dataframe.loc[dataframe[column] == c]
            temp = temp.sample(n=histogram["size"].min())
            df_list.append(temp)
        return pd.concat(df_list)

    def create_datasets(
        self,
        train_size: float,
        validation_size: float,
        test_size: float,
        ensure_distinct_users_per_dataset: bool = True,
        balance_samples: bool = True,
        activities_remap: Dict[int, int] = None,
        seed: int = None,
        use_tqdm: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test datasets.

        Parameters
        ----------
        train_size : float
            Fraction of samples to training dataset.
        validation_size : float
            Fraction of samples to validation dataset.
        test_size : float
            Fraction of samples to test dataset.
        ensure_distinct_users_per_dataset : bool
            If True, ensure that samples from an user do not belong to distinct
            datasets (the default is True).
        balance_samples : bool
            If True, the datasets will have the same number of samples per
            class. The number of samples will be reduced to the class with the
            minor number of samples (the default is True).
        activities_remap : Dict[int, int]
            A dictionaty used to replace a label from one class to another.
        seed : int
            The random seed (the default is None).
        use_tqdm : bool
            If must use tqdm as iterator (the default is True).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple with the train, validation and test dataframes.

        """

        assert np.isclose(
            sum([train_size, validation_size, test_size]), 1.0
        ), "The sizes must sum up to 1"
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)

        df = self.get_full_df(use_tqdm=use_tqdm)
        users = df["user"].unique()
        activities = df["activity code"].unique()

        train, validation, test = self.train_test_split(
            df=df,
            users=users,
            activities=activities,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            ensure_distinct_users_per_dataset=ensure_distinct_users_per_dataset,
            seed=seed,
        )

        if ensure_distinct_users_per_dataset:
            if (
                not self.check_if_unique_per_df(train, [validation, test])
                or not self.check_if_unique_per_df(validation, [train, test])
                or not self.check_if_unique_per_df(test, [validation, train])
            ):
                raise DatasetSplitError(
                    "Samples from the same user belongs to different dataset splits."
                )

        if activities_remap is not None:
            train["activity code"].replace(activities_remap, inplace=True)
            validation["activity code"].replace(activities_remap, inplace=True)
            test["activity code"].replace(activities_remap, inplace=True)

        # balance datasets!
        if balance_samples:
            train = self.balance_dataset_to_minimum(train)
            validation = self.balance_dataset_to_minimum(validation)
            test = self.balance_dataset_to_minimum(test)

        # reset indexes
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)

        return train, validation, test

    def __str__(self) -> str:
        return f"Dataset generator: time_window={self.time_window}, overlap={self.window_overlap}"

    def __repr__(self) -> str:
        return str(self)
