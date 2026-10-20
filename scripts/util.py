import math
import os
import time
import warnings
import re

import joblib
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error, log_loss)
import tensorflow
from sklearn.model_selection import KFold
import importlib
import ast

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'loss', 'mae', 'auc', 'time']


def save_plot(path):
    if not os.path.exists(path):
        plt.savefig(path)


def measure_time(start=None):
    end = time.time()
    return float(f'{(end - start):.1f}') if start is not None else float(f'{end:.1f}')


def format_float(number):
    return float(f'{number:.4f}')


def format_percent(number):
    return float(f'{(number * 100):.2f}')


def int_from_str(text):
    return int(re.search(r"\d?\d+", text).group())


def train_test_from(x, y, fraction=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=fraction, random_state=42)
    return x_train, y_train, x_test, y_test


def evaluate(probabilities, labels, training_time=None):
    predictions = [np.argmax(i) for i in probabilities]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    loss = log_loss(labels, probabilities)
    y_test_categorical = tensorflow.keras.utils.to_categorical(labels, num_classes=probabilities.shape[1])
    mae = mean_absolute_error(y_test_categorical, probabilities)
    try:
        auc = roc_auc_score(y_test_categorical, probabilities, multi_class='ovr')
    except Exception:
        auc = None
    metrics = {'accuracy': format_float(accuracy), 'precision': format_float(precision),
               'recall': format_float(recall), 'f1': format_float(f1),
               'loss': format_float(loss), 'mae': format_float(mae),
               'auc': format_float(auc) if auc is not None else 0.0}
    if training_time is not None:
        metrics['time'] = training_time
    return metrics


def use_pca(train, test, dims, reverse=False, pca_export_path=None):
    pca = PCA(n_components=dims)
    train = pca.fit_transform(train)
    variance = np.sum(pca.explained_variance_ratio_)
    test = pca.transform(test)
    if reverse:
        train = pca.inverse_transform(train)
        test = pca.inverse_transform(test)
    if pca_export_path is not None:
        joblib.dump(pca, pca_export_path)
    return train, test, variance


def capital_letters(text):
    return ''.join(re.findall(r'[A-Z]', text))


def crop_pos_class_name(text):
    text = str(text)
    match = re.search(r"(?<=\.)\w+(?='?>)", text)
    return match.group() if match else text


def best_rect_dims_of_(number):
    first, second = 1, number
    for i in range(2, int(math.sqrt(number))):
        division = int(number // i)
        if i * division == number:
            first, second = i, division
    return first, second


def cross_validate(args):
    model_usage_function, data, params, pca_dims, splits = args
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)
    x, y = data
    results = []
    for train_index, test_index in kf.split(x):
        if pca_dims:
            x_train, x_test, variance = use_pca(x.iloc[train_index], x.iloc[test_index], pca_dims)
            split_data = (x_train, y.iloc[train_index], x_test, y.iloc[test_index])
        else:
            split_data = x.iloc[train_index], y.iloc[train_index], x.iloc[test_index], y.iloc[test_index]

        results.append(model_usage_function((split_data, params)))
    results = pd.DataFrame(results)
    tmp = dict()
    for col in results.columns:
        try:
            val = results[col].mean() if not isinstance(results[col][0], str) else results[col][0]
            tmp[col] = val
        except Exception:
            tmp[col] = str(results[col][0])
    results = tmp
    return results


def df_to_param_dicts(df: pd.DataFrame):
    df = df.drop('Unnamed: 0', axis=1)
    numerical_cols = df.select_dtypes(include=np.number).columns
    numerical_cols = set(numerical_cols).difference(metrics_names)
    cols = list(set(df.columns).difference(metrics_names))
    dicts = []
    for i, row in df.iterrows():
        tmp = dict()
        for col in cols:
            if pd.isna(row[col]):
                tmp[col] = None  # None values
            elif col not in numerical_cols:
                try:
                    val = row[col].replace("<class '", "").replace("'>", "")
                    module_path, class_name = val.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    tmp[col] = getattr(module, class_name)  # classes
                except (ValueError, ImportError, AttributeError):
                    try:
                        tmp[col] = [int(x) for x in ast.literal_eval(row[col])]  # arrays
                    except (ValueError, SyntaxError):
                        tmp[col] = row[col]  # strings
            else:
                tmp[col] = row[col]  # numerical values
        dicts.append(tmp)
    return dicts


def optional_int(val):
    return int(val) if val else None
