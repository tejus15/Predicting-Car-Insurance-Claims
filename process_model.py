import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_auc_score, log_loss, confusion_matrix

"""
process_model.py

Author: Harsha Gurram
Date: Fall 2024
Course: Applied Machine Learning
Project: Insurance Classificaion

Description:
Allows you to run and effectively store models for a given dataset.

"""

# Your code starts here
class Processor:
    def __init__(self, data_df, target_col, quant_cols, qual_cols, random_state=42):
        self.data_df = data_df
        self.target_col = target_col
        self.quant_cols = quant_cols
        self.qual_cols = qual_cols
        self.random_state = random_state
        self.model_store = {}

    def train_test_split(self, test_size=0.2, stratify_by=None):
        """
        Splits the dataset into training and testing sets. 
        
        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        stratify_by (str): The column to stratify the split by.
        """

        X = self.data_df[self.quant_cols + self.qual_cols]
        X_enc = pd.get_dummies(X, columns=self.qual_cols, drop_first=True)
        y = self.data_df[self.target_col]

        if stratify_by:
            stratify_key = pd.Series(list(zip(y, X[stratify_by])))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_enc, 
                y, 
                test_size=test_size, 
                random_state=self.random_state, 
                stratify=stratify_key
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_enc, 
                y, 
                test_size=test_size, 
                random_state=self.random_state
            )

        return None

    def data_standarization(self, processor='StandardScaler'):
        """
        Applies standardization to the quantiative columns in the dataset. 
        Assumes that the dataset has been processed to help models deal with qualitative columns
        
        Parameters:
        quant_cols (list): List of column names to be standardized.
        processor (str): The type of scaler to use for standardization. 
                         Accepts two options:
                         - 'StandardScaler': Standardizes features by removing the mean and scaling to unit variance.
                         - 'MinMaxScaler': Transforms features by scaling each feature to a given range (default is 0 to 1).

        Returns:
        DataFrame: A DataFrame with the standardized columns for specified columns.
        """

        if processor == 'StandardScaler':
            scaler = StandardScaler()
        elif processor == 'MinMaxScaler':
            scaler = MinMaxScaler()

        self.X_train[self.quant_cols] = scaler.fit_transform(self.X_train[self.quant_cols])
        self.X_test[self.quant_cols] = scaler.transform(self.X_test[self.quant_cols])

        return None

    def oversample_data(self):
        """
        Oversamples the minority class in the dataset. 
        
        Parameters:
        target_col (str): The target column to oversample.

        Returns:
        DataFrame: A DataFrame with the oversampled dataset.
        """

        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        return None
    
    def run_model(self, model_name, model_obj, parameter=None, cv=5, scoring='accuracy'):

        if parameter:
            grid_search = GridSearchCV(estimator=model_obj, param_grid=parameter, cv=cv, scoring=scoring)
            grid_search.fit(self.X_train, self.y_train)
            mod_model_obj = grid_search.best_estimator_
            self.y_pred = mod_model_obj.predict(self.X_test)
        else:
            mod_model_obj = model_obj
            mod_model_obj.fit(self.X_train, self.y_train)
            self.y_pred = mod_model_obj.predict(self.X_test)

        self.model_store[model_name] = {
            'model': mod_model_obj,
            'y_pred': self.y_pred,
            'results': {}
        }

        self.model_store[model_name]["results"]["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        self.model_store[model_name]["results"]["precision"] = precision_score(self.y_test, self.y_pred, zero_division=0)
        self.model_store[model_name]["results"]["recall"] = recall_score(self.y_test, self.y_pred)
        self.model_store[model_name]["results"]["f1-score"] = f1_score(self.y_test, self.y_pred)
        self.model_store[model_name]["results"]["auc"] = roc_auc_score(self.y_test, self.y_pred)
        self.model_store[model_name]["results"]["log loss"] = log_loss(self.y_test, self.y_pred)
        self.model_store[model_name]["results"]["confusion matrix"] = confusion_matrix(self.y_test, self.y_pred)

        return None
    
    def compute_results(self, y_actual, y_pred):
        results = {
            'accuracy': accuracy_score(y_actual, y_pred),
            'precision': precision_score(y_actual, y_pred, zero_division=0),
            'recall': recall_score(y_actual, y_pred),
            'f1-score': f1_score(y_actual, y_pred),
            'auc': roc_auc_score(y_actual, y_pred),
            'log loss': log_loss(y_actual, y_pred),
            'confusion matrix': confusion_matrix(y_actual, y_pred)
        }

        return results
    
    def get_results(self):

        results_df = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1-score", "auc", "log loss"])

        for model_name, model_info in self.model_store.items():
            results = model_info["results"]
            results_df.loc[model_name] = [
            results["accuracy"],
            results["precision"],
            results["recall"],
            results["f1-score"],
            results["auc"],
            results["log loss"]
            ]

        return results_df
    
    def dump_pkl(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class PartProcessor(Processor):
    def __init__(self, data_df, target_col, strat_col, quant_cols, qual_cols, min_n_parts = 1000, random_state=42):
        self.stratify_by = strat_col
        self.min_n_parts = min_n_parts
        self.oversample_flag = False
        self.model_results_by_stratify = {}
        super().__init__(data_df, target_col, quant_cols, qual_cols, random_state)

    def train_test_split(self, test_size=0.2):
        temp_X = self.data_df[self.quant_cols + self.qual_cols]
        temp_y = self.data_df[self.target_col]
        temp_stratify_key = pd.Series(list(zip(self.data_df[self.target_col], temp_X[self.stratify_by])))
        self.unenc_X_train, self.unenc_X_test, self.unenc_y_train , temp = train_test_split(
            temp_X, 
            temp_y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=temp_stratify_key
        )
        super().train_test_split(test_size, self.stratify_by)
        
    def data_standarization(self, processor='StandardScaler'):
        super().data_standarization(processor)

    def compute_results(self, y_pred, y_actual):
        return super().compute_results(y_pred, y_actual)

    def oversample_data(self):
        self.oversample_flag = True
    
    def get_results(self):
        return super().get_results()
    
    def model_partitions(self):
        stratify_counts = self.unenc_X_train[self.stratify_by].value_counts()
        stratify_list = []
        temp_list = []

        for stratify_value, count in stratify_counts.items():
            if count <= self.min_n_parts:
                temp_list.append(stratify_value)
            else:
                stratify_list.append(stratify_value)

        if temp_list:
            stratify_list.append(temp_list)

        return stratify_list
    
    def run_model(self, model_name, model_obj, parameter=None, cv=5, scoring='accuracy', n_jobs=-1):
        stratify_list = self.model_partitions()
        y_pred_global = pd.Series(np.zeros(self.y_test.shape), index=self.y_test.index)
                
        for stratify_value in stratify_list: 

            current_fold = stratify_value if isinstance(stratify_value, str) else "Other"

            if current_fold == "Other":
                X_train_temp = self.X_train[self.unenc_X_train[self.stratify_by].isin(stratify_value)]
                y_train_temp = self.y_train[self.unenc_X_train[self.stratify_by].isin(stratify_value)]
                X_test_temp = self.X_test[self.unenc_X_test[self.stratify_by].isin(stratify_value)]
                test_indices = self.X_test[self.unenc_X_test[self.stratify_by].isin(stratify_value)].index
            else:
                X_train_temp = self.X_train[self.unenc_X_train[self.stratify_by] == stratify_value]
                y_train_temp = self.y_train[self.unenc_X_train[self.stratify_by] == stratify_value]
                X_test_temp = self.X_test[self.unenc_X_test[self.stratify_by] == stratify_value]
                test_indices = self.X_test[self.unenc_X_test[self.stratify_by] == stratify_value].index

            if self.oversample_flag:
                smote = SMOTE(random_state=self.random_state)
                X_train_temp, y_train_temp = smote.fit_resample(X_train_temp, y_train_temp)

            if parameter:
                grid_search = GridSearchCV(estimator=model_obj, param_grid=parameter, cv=cv, scoring=scoring, n_jobs=n_jobs)
                grid_search.fit(X_train_temp, y_train_temp)
                mod_model_obj = grid_search.best_estimator_
                y_pred_temp = mod_model_obj.predict(X_test_temp)
            else:
                mod_model_obj = model_obj
                mod_model_obj.fit(X_train_temp, y_train_temp)
                y_pred_temp = mod_model_obj.predict(X_test_temp)

            # Also store the results for each car model in a separate df to see if one model works better 
            # for a specific car model vs another ml model.

            if current_fold not in self.model_results_by_stratify:
                self.model_results_by_stratify[current_fold] = {}
            
            self.model_results_by_stratify[current_fold][model_name] = {
                'results': self.compute_results(self.y_test[test_indices], y_pred_temp),
                'model': mod_model_obj,
                'y_actual': self.y_test[test_indices],
                'y_pred': y_pred_temp
            }

            y_pred_global[test_indices] = y_pred_temp

        self.model_store[model_name] = {
            'y_pred': y_pred_global,
            'results': self.compute_results(self.y_test, y_pred_global)
        }

        return None
    
    def dump_pkl(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

