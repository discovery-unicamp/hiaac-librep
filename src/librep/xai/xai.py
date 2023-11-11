from pathlib import Path
from typing import List, Tuple, Dict, Any


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from librep.datasets.multimodal.operations import (
    DatasetFitter,
    PandasMultiModalDataset,
)

# import xAI techniques
import shap
from lime import lime_tabular

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


############################################################################################################
# Function to load the dataset
############################################################################################################
def load_dataset(
    dataset_name: str,
    reduce_on: str,
    normalization: str = None,
    path: Path = Path("../reducer_experiments/results/execution/transformed_data"),
) -> Tuple[PandasMultiModalDataset, PandasMultiModalDataset]:
    """This function loads the dataset from the path. In particular, it loads the train and test files from the path:
    results/execution/output_files/reduced_data/{dataset_name}-{reduce_on}.
    This directory contains the preprocessed data for the interested experiment.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset
    reduce_on: str
        The name of the modality on which the dataset is reduced
    normalization: str
        The name of the normalization technique to apply to the dataset
    path: Path
        The path where the dataset is stored

    Returns
    -------
    train: Dataset
        The train dataset
    test: Dataset
        The test dataset
    """
    path = path / f"{dataset_name}-{reduce_on}"

    # Let's read files from the directory path
    with open(path / "train.pkl", "rb") as f:
        train = pickle.load(f)

    with open(path / "test.pkl", "rb") as f:
        test = pickle.load(f)

    # Let's normalize the dataset
    train, test = normalize_dataset(train, test, normalization)

    return train, test


def normalize_dataset(
    train: PandasMultiModalDataset,
    test: PandasMultiModalDataset,
    normalization: str = None,
) -> Tuple[PandasMultiModalDataset, PandasMultiModalDataset]:
    """This function normalizes the dataset.

    Parameters
    ----------
    train: Dataset
        The train dataset
    test: Dataset
        The test dataset
    normalization: str
        The name of the normalization technique to apply to the dataset
        If None, no normalization is applied
        If "MinMaxScaler", MinMaxScaler is applied
        If "StandardScaler", StandardScaler is applied

    Returns
    -------
    train: Dataset
        The normalized train dataset
    test: Dataset
        The normalized test dataset
    """

    if normalization == None:
        pass
    elif normalization == "MinMaxScaler":
        scaler = MinMaxScaler()
        scaler.fit(train.X)
        train.X = scaler.transform(train.X)
        test.X = scaler.transform(test.X)
    elif normalization == "StandardScaler":
        scaler = StandardScaler()
        scaler.fit(train.X)
        train.X = scaler.transform(train.X)
        test.X = scaler.transform(test.X)

    return train, test


############################################################################################################
# Functions to train the models
# Random Forest
# SVM
# KNN
# Decision Tree
############################################################################################################
def train_rf(train: PandasMultiModalDataset) -> RandomForestClassifier:
    """This function trains a Random Forest classifier on the train dataset.

    Parameters
    ----------
    train: Dataset
        The train dataset

    Returns
    -------
    model: RandomForestClassifier
        The trained Random Forest classifier
    """

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    DatasetFitter(model, use_y=True)(train)

    return model


def train_svm(train: PandasMultiModalDataset) -> SVC:
    """This function trains a SVM classifier on the train dataset.

    Parameters
    ----------
    train: Dataset
        The train dataset

    Returns
    -------
    model: SVC
        The trained SVM classifier
    """
    model = SVC(random_state=42, probability=True)
    DatasetFitter(model, use_y=True)(train)

    return model


def train_knn(train: PandasMultiModalDataset) -> KNeighborsClassifier:
    """This function trains a KNN classifier on the train dataset.

    Parameters
    ----------
    train: Dataset
        The train dataset

    Returns
    -------
    model: KNeighborsClassifier
        The trained KNN classifier
    """
    model = KNeighborsClassifier(n_neighbors=5)
    DatasetFitter(model, use_y=True)(train)

    return model


def train_dt(train: PandasMultiModalDataset) -> DecisionTreeClassifier:
    """This function trains a Decision Tree classifier on the train dataset.

    Parameters
    ----------
    train: Dataset
        The train dataset

    Returns
    -------
    model: DecisionTreeClassifier
        The trained Decision Tree classifier
    """
    model = DecisionTreeClassifier(random_state=42)
    DatasetFitter(model, use_y=True)(train)

    return model


############################################################################################################
# Functions to calculate the feature importance using SHAP
############################################################################################################
def calc_shap_values_tree(model, test: PandasMultiModalDataset) -> np.ndarray:
    """This function calculates the shap values for each sample in the test dataset

    Parameters
    ----------
    model: sklearn model based on trees
        The trained model
    test: Dataset
        The test dataset

    Returns
    -------
    shap_values: np.ndarray
        The shap values for each sample in the test dataset
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test.X)

    return shap_values


def calc_shap_values(model, test: PandasMultiModalDataset) -> np.ndarray:
    """This function calculates the shap values for each sample in the test dataset

    Parameters
    ----------
    model: machine learning model
        The trained model
    test: Dataset
        The test dataset
    """
    explainer = shap.KernelExplainer(model.predict_proba, test.X)
    shap_values = explainer.shap_values(test.X, nsamples=100)

    return shap_values


############################################################################################################
# Functions to filter the SHAP values
############################################################################################################
def shap_values_per_feature(
    shap_values, activities: List[int], num_features: int = 24
) -> pd.DataFrame:
    """This function calculates the shap values for each feature, for each activity.
    For each activity, the shap values from a subset are the absolute average of the shap values of all samples in the subset.
    After that, the feature importance for each feature are the sum of the shap values of all activities.

    Parameters
    ----------
    shap_values: np.ndarray
        The shap values for each sample in the test dataset
    activities: List[int]
        The list of activities
    num_features: int
        The number of features

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the shap values for each feature
    """
    fi = {
        f"feature {i}": np.sum(
            [
                np.mean(np.abs(shap_values[j][:, i]))
                for j, activity in enumerate(activities)
            ]
        )
        for i in range(num_features)
    }

    df = pd.DataFrame(fi, index=[0])
    return df


def shap_values_per_class(shap_values, activities: List[int]) -> pd.DataFrame:
    """This function calculates the shap values for each feature, for each activity.
    For each activity, the shap values from a subset are the absolute average of the shap values of all samples in the subset.

    Parameters
    ----------
    shap_values: np.ndarray
        The shap values for each sample in the test dataset
    activities: List[int]
        The list of activities

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the shap values for each feature
    """
    keys = ["activity"] + [f"feature {i}" for i in range(24)]

    fi = {key: None for key in keys}
    fis = []
    for j, activity in enumerate(activities):
        fi["activity"] = activity
        for i in range(24):
            fi[f"feature {i}"] = np.mean(np.abs(shap_values[j][:, i]))
        fis.append(fi.copy())

    df = pd.DataFrame(fis)
    return df


############################################################################################################
# Functions to calculate the feature importance using LIME
############################################################################################################
def calc_lime_values(
    model,
    test: PandasMultiModalDataset,
    train: PandasMultiModalDataset,
    standartized_codes: Dict[int, str],
) -> List[Dict[str, Any]]:
    """This function calculates the lime values for each sample in the test dataset and store in a dictionary
    the class of the sample, the predicted class by lime, the feature importance and the explainer object created by lime.

    Parameters
    ----------
    model: machine learning model
        The trained model
    test: Dataset
        The test dataset
    train: Dataset
        The train dataset
    standartized_codes: Dict[int, str]
        The dictionary containing the mapping between the activity codes and the activity names

    Returns
    -------
    lime_values: List[Dict[str, Any]]
        The list of dictionaries containing the lime values for each sample in the test dataset
    """
    # This function calculates the lime values for each sample in the test dataset
    lime_values = []
    num_features = train.X.shape[1]
    features_names = [f"feature_{i}" for i in range(num_features)]
    explainer = lime_tabular.LimeTabularExplainer(
        train.X,
        feature_names=features_names,
        class_names=standartized_codes.values(),
        discretize_continuous=False,
    )
    for sample, y in tqdm(zip(test.X, test.y)):
        exp = explainer.explain_instance(
            sample,
            model.predict_proba,
            num_features=num_features,
            top_labels=1,
        )

        lime_value = {
            "True class": y,
            "LIME prediction": list(exp.as_map().keys())[0],
            "Model prediction": model.predict([sample])[0],
            "Lime values": exp.as_map(),
            "Explainer": exp,
        }

        lime_values.append(lime_value)

    return lime_values


############################################################################################################
# Functions to filter the LIME values
############################################################################################################
def lime_values_per_class(
    lime_values: List[Dict[str, Any]],
    dataset: str,
    reduce: str,
    model_name: str,
    activities: List[int],
    standartized_codes: Dict[int, str],
    num_features: int = 24,
    remove_misclassified: bool = True,
) -> pd.DataFrame:
    """This function calculates the lime values for each feature, for each activity. For each activity,
    the lime values from a subset are the absolute average of the feature importances of all samples in the subset.
    After that, the feature importance for each feature are the sum of the lime values of all activities.

    Parameters
    ----------
    lime_values: List[Dict[str, Any]]
        The list of dictionaries containing the lime values for each sample in the test dataset
    dataset: str
        The name of the dataset
    reduce: str
        The name of the modality on which the dataset is reduced
    model_name: str
        The name of the model
    activities: List[int]
        The list of activities
    standartized_codes: Dict[int, str]
        The dictionary containing the mapping between the activity codes and the activity names
    num_features: int
        The number of features

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the feature importance for each feature for each activity by lime
    """

    dfs = {activity: None for activity in activities}

    fis = {activity: [] for activity in activities}
    for j, lime_value in enumerate(lime_values):
        # Calculate the feature importance for the sample
        activity = lime_value["True class"]
        lime_predict = lime_value["LIME prediction"]
        sample = lime_value["Lime values"][lime_predict]
        sample = sorted(sample)
        sample = np.array(sample)
        fi = np.abs(sample[:, 1])
        model_predict = lime_value["Model prediction"]
        if remove_misclassified:
            fis[activity].append(fi)
        else:
            # Check if the model predicted correctly the sample
            if model_predict == activity:
                fis[activity].append(fi)

    # Let's calculate the average of the feature importance for each activity
    columns = ["Classifier", "Dataset", "reduce on", "activity"] + [
        f"feature_{i}" for i in range(num_features)
    ]
    for activity in activities:
        fi_class = np.array(fis[activity])
        fi_class = np.mean(fi_class, axis=0)
        fi_class = fi_class.tolist()
        data = [model_name, dataset, reduce, standartized_codes[activity]] + fi_class
        df = pd.DataFrame([data], columns=columns)
        dfs[activity] = df

    # The dataframe containing the global feature importance for each class of the dataset
    df = pd.concat([sub_df for sub_df in dfs.values()])

    return df


def lime_values_per_feature(
    lime_values: List[Dict[str, Any]],
    dataset: str,
    reduce: str,
    model_name: str,
    activities: List[int],
    standartized_codes: Dict[int, str],
    num_features: int = 24,
) -> pd.DataFrame:
    """This function calculates the lime values for each feature, for each activity. For each activity,
    the lime values from a subset are the absolute average of the feature importances of all samples in the subset.
    After that, the feature importance for each feature are the sum of the lime values of all activities.

    Parameters
    ----------
    lime_values: List[Dict[str, Any]]
        The list of dictionaries containing the lime values for each sample in the test dataset
    dataset: str
        The name of the dataset
    reduce: str
        The name of the modality on which the dataset is reduced
    model_name: str
        The name of the model
    activities: List[int]
        The list of activities
    standartized_codes: Dict[int, str]
        The dictionary containing the mapping between the activity codes and the activity names
    num_features: int
        The number of features

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing the feature importance for each feature for each activity by lime
    """

    df = lime_values_per_class(
        lime_values,
        dataset,
        reduce,
        model_name,
        activities,
        standartized_codes,
        num_features,
    )
    # Let's remove the columns that are not features
    df = df.drop(columns=["Classifier", "Dataset", "reduce on", "activity"])

    # Let's sum all lines
    df = df.sum(axis=0).to_frame().T

    return df.reset_index(drop=True)


############################################################################################################
# Functions to calculate the feature importance using Oracle thecnique
############################################################################################################
def calc_oracle_values(
    classifier: str,
    dataset: str,
    reduce: str,
    latent_dim: int = 24,
    columns_to_remove: List[int] = [],
    normalization: str = None,
    data_path: str = "../reducer_experiments/results/execution/transformed_data",
):
    """This function calculates the oracle values for each feature.
        The oracle values are calculated by removing each feature from the dataset and then training a classifier on the reduced dataset,
        so that the accuracy of the classifier is calculated
        on the test dataset. The oracle values are calculated as 1 - accuracy.

    Parameters
    ----------
    classifier: str
        The name of the classifier
    dataset: str
        The name of the dataset
    reduce: str
        The name of the modality on which the dataset is reduced
    latent_dim: int
        The number of latent dimensions, that is, the number of features
    columns_to_remove: List[int]
        The list of columns to remove from the dataset, if needed

    Returns
    -------
    fis: np.ndarray
        The oracle values for each feature
    """

    accuracies = []
    for dim in range(latent_dim):
        train, test = load_dataset(
            dataset, reduce, normalization=normalization, path=Path(data_path)
        )
        if columns_to_remove != []:
            train.X = np.delete(train.X, columns_to_remove, axis=1)
            test.X = np.delete(test.X, columns_to_remove, axis=1)
        # Let's remove the column dim from the train and test dataset
        train.X = np.delete(train.X, dim, axis=1)
        test.X = np.delete(test.X, dim, axis=1)

        model = (
            train_rf(train)
            if classifier == "Radom Forest"
            else train_svm(train)
            if classifier == "SVM"
            else train_knn(train)
            if classifier == "KNN"
            else train_dt(train)
        )
        accuracy = model.score(test.X, test.y)
        accuracies.append(accuracy)

    fis = 1 - np.array(accuracies)
    return fis, accuracies
