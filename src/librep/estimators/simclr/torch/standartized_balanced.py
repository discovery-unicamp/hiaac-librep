import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class StandardizedBalancedDataset:
    def __init__(self, data_folder, sensors=["accel"]):
        self.data_folder = data_folder
        self.sensors = sensors
        self.scaler = StandardScaler()

    
    def load_data(self, filename):
        df = pd.read_csv(os.path.join(self.data_folder, filename))
        #print(df.columns)
        X, y = self.get_data(df)
        return X, y

    def get_data(self, df):
        patterns = []
        for sensor in self.sensors:
            patterns.extend([f"{sensor}-{axis}" for axis in ["x", "y", "z"]])

        selected_columns = pd.DataFrame()

        for pattern in patterns:
            matching_columns = df.filter(like=pattern)
            selected_columns = pd.concat([selected_columns, matching_columns], axis=1)

        #label_columns = pd.get_dummies(df['standard activity code']).to_numpy()
        label_columns=df['standard activity code']

        return selected_columns.to_numpy(), label_columns

    def preprocess(self, X, normalize=True, resized=True):
        if normalize:
            X = self.scaler.fit_transform(X)
        if resized:
            X = self.resize_data(X)
        return X

    def resize_data(self, data):
        submatrix_rows = 60
        submatrix_cols = len(self.sensors) * 3
        result_matrices = []

        for row in data:
            row_matrices = np.split(row, len(row) // submatrix_cols)
            result_matrices.append(row_matrices)
        final_array = np.array(result_matrices)
        return final_array

    def get_data_train(self, normalize_data=True, resize_data=True):
        X_train, y_train = self.load_data('train.csv')
        X_train = self.preprocess(X_train, normalize_data, resize_data)
        return X_train, y_train
    
    def get_data_test(self, normalize_data=True, resize_data=True):
        X_test, y_test = self.load_data('test.csv')
        X_test = self.preprocess(X_test, normalize_data, resize_data)
        return X_test, y_test
    
    def get_data_val(self, normalize_data=True, resize_data=True):
        X_val, y_val = self.load_data('validation.csv')
        X_val = self.preprocess(X_val, normalize_data, resize_data)
        return X_val, y_val
    
    def get_all_data(self, normalize_data=True, resize_data=True):
        X_train, y_train = self.get_data_train(normalize_data=normalize_data, resize_data=resize_data)
        X_test, y_test = self.get_data_test(normalize_data=normalize_data, resize_data=resize_data)
        X_val, y_val = self.get_data_val(normalize_data=normalize_data, resize_data=resize_data)
        return X_train, y_train,X_test, y_test,X_val, y_val
