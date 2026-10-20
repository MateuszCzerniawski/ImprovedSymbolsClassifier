import itertools
import os
import random
import re
import sys

import numpy as np
import pandas as pd

import util
import multiprocessing as mp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from scripts import data_manipulation as dm

criteria = ['gini', 'entropy', 'log_loss']
max_depths = [None, 10, 20, 30]
n_estimators = [50, 100, 125, 150]
min_splits = [2, 5, 10]
min_leaves = [2, 4, 5]
max_features = ['sqrt', 'log2', None]
learning_rates = [0.01, 0.1, 0.3]
min_child_weight = [1, 5, 10]
reg = [('l1', 0.1), ('l1', 0.5), ('l1', 1), ('l2', 1), ('l2', 10), ('l2', 100)]

model_hiperparameters = {
    'DecisionTreeClassifier': [criteria, max_depths, min_splits, min_leaves, max_features],
    'ExtraTreesClassifier': [criteria, n_estimators, max_depths, min_splits, min_leaves, max_features],
    'RandomForestClassifier': [criteria, n_estimators, max_depths, min_splits, min_leaves, max_features],
    'XGBClassifier': [n_estimators, max_depths, learning_rates, min_child_weight, reg]
}
model_short = {
    'DT': 'DecisionTreeClassifier',
    'ET': 'ExtraTreesClassifier',
    'RF': 'RandomForestClassifier',
    'XGB': 'XGBClassifier'
}


def combine_tests(model):
    if model not in model_hiperparameters:
        model = model_short[model]
    tests = [list(i) for i in itertools.product(*model_hiperparameters[model])]
    if model == 'DecisionTreeClassifier':
        return [{'criterion': test[0], 'max_depth': test[1], 'min_samples_split': test[2],
                 'min_samples_leaf': test[3], 'max_features': test[4]} for test in tests]
    if model == 'ExtraTreesClassifier' or model == 'RandomForestClassifier':
        return [{'criterion': test[0], 'n_estimators': test[1], 'max_depth': test[2], 'min_samples_split': test[3],
                 'min_samples_leaf': test[4], 'max_features': test[5]} for test in tests]
    if model == 'XGBClassifier':
        return [{'n_estimators': test[0], 'max_depth': test[1], 'learning_rate': test[2],
                 'min_child_weight': test[3], 'regularizer': test[4][0], 'reg_val': test[4][1]} for test in tests]


def fit_evaluate(model, data, params):
    x_train, y_train, x_test, y_test = data
    start = util.measure_time()
    model.fit(x_train, y_train)
    end = util.measure_time(start)
    predictions = model.predict_proba(x_test)
    evaluation = util.evaluate(predictions, y_test, end)
    for key, value in evaluation.items():
        params[key] = value
    return params


def use_decision_tree(args):
    data, params = args
    model = DecisionTreeClassifier(
        criterion=params['criterion'], max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features']
    )
    return fit_evaluate(model, data, params)


def use_extra_tree(args):
    data, params = args
    model = ExtraTreesClassifier(
        criterion=params['criterion'], n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features']
    )
    return fit_evaluate(model, data, params)


def use_random_forest(args):
    data, params = args
    model = RandomForestClassifier(
        criterion=params['criterion'], n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features']
    )
    return fit_evaluate(model, data, params)


def use_xgboost(args):
    data, params = args
    if params['regularizer'] == 'l1':
        model = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                              learning_rate=params['learning_rate'],
                              min_child_weight=params['min_child_weight'], reg_alpha=params['reg_val'])
    else:
        model = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                              learning_rate=params['learning_rate'],
                              min_child_weight=params['min_child_weight'], reg_lambda=params['reg_val'])
    return fit_evaluate(model, data, params)


def use_decision_tree_seed(args):
    data, params = args
    model = DecisionTreeClassifier(
        criterion=params['criterion'], max_depth=util.optional_int(params['max_depth']),
        min_samples_split=util.optional_int(params['min_samples_split']), min_samples_leaf=util.optional_int(params['min_samples_leaf']),
        max_features=params['max_features'],
        random_state=42
    )
    return fit_evaluate(model, data, params)


def use_extra_tree_seed(args):
    data, params = args
    model = ExtraTreesClassifier(
        criterion=params['criterion'], n_estimators=util.optional_int(params['n_estimators']),
        max_depth=util.optional_int(params['max_depth']), min_samples_split=util.optional_int(params['min_samples_split']),
        min_samples_leaf=util.optional_int(params['min_samples_leaf']), max_features=params['max_features'],
        random_state=42
    )
    return fit_evaluate(model, data, params)


def use_random_forest_seed(args):
    data, params = args
    model = RandomForestClassifier(
        criterion=params['criterion'], n_estimators=util.optional_int(params['n_estimators']),
        max_depth=util.optional_int(params['max_depth']),
        min_samples_split=util.optional_int(params['min_samples_split']),
        min_samples_leaf=util.optional_int(params['min_samples_leaf']), max_features=params['max_features'],
        random_state=42
    )
    return fit_evaluate(model, data, params)


def use_xgboost_seed(args):
    data, params = args
    if params['regularizer'] == 'l1':
        model = XGBClassifier(n_estimators=util.optional_int(params['n_estimators']), max_depth=util.optional_int(params['max_depth']),
                              learning_rate=params['learning_rate'],
                              min_child_weight=util.optional_int(params['min_child_weight']), reg_alpha=params['reg_val'],
                              random_state=42)
    else:
        model = XGBClassifier(n_estimators=util.optional_int(params['n_estimators']), max_depth=util.optional_int(params['max_depth']),
                              learning_rate=params['learning_rate'],
                              min_child_weight=util.optional_int(params['min_child_weight']), reg_lambda=params['reg_val'],
                              random_state=42)
    return fit_evaluate(model, data, params)


def conduct_tests(model, data, path, tests=None, tests_checkpoint=25, checkpoint_dir='../tmp'):
    function = {
        'DecisionTreeClassifier': use_decision_tree,
        'ExtraTreesClassifier': use_extra_tree,
        'RandomForestClassifier': use_random_forest,
        'XGBClassifier': use_xgboost
    }[model]
    if tests is None:
        tests = combine_tests(model)
    tests = [(data, i) for i in tests]
    checkpoint_file = f'{checkpoint_dir}/{model}.csv'
    last = 0
    if os.path.exists(checkpoint_file):
        last = pd.read_csv(checkpoint_file).shape[0]
        print(f'Last detected point is {last}, skipping...')
    ranges = [(i, i + tests_checkpoint if i + tests_checkpoint < len(tests) else len(tests)) for i in
              range(last, len(tests), tests_checkpoint)]
    for checkpoint_start, checkpoint_end in ranges:
        print(f'Checkpoint {checkpoint_start}-{checkpoint_end}')
        try:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = [i for i in pool.map(function, tests[checkpoint_start:checkpoint_end])]
                results = pd.DataFrame(results)
                if os.path.exists(checkpoint_file):
                    checkpoint_results = pd.read_csv(checkpoint_file, index_col=False)
                    results = pd.concat([checkpoint_results, results], ignore_index=True)
                results.to_csv(checkpoint_file, index=False)
        finally:
            pool.close()
            pool.join()
    checkpoint_data = pd.read_csv(checkpoint_file)
    if checkpoint_data.shape[0] == len(tests):
        checkpoint_data.to_csv(path)
        os.remove(checkpoint_file)
    else:
        print(f'Something wnet wrong, checkpoint has {checkpoint_data.shape[0]} records of {len(tests)}')


def worker_init(seed):
    np.random.seed(seed)
    random.seed(seed)


def conduct_tests_cv(model, data, path, tests=None, pca_dims=None, splits=5, tests_checkpoint=25,
                     checkpoint_dir='../tmp'):
    worker_init(42)
    function = {
        'DecisionTreeClassifier': use_decision_tree_seed,
        'ExtraTreesClassifier': use_extra_tree_seed,
        'RandomForestClassifier': use_random_forest_seed,
        'XGBClassifier': use_xgboost_seed
    }[model]
    if tests is None:
        tests = combine_tests(model)
    tests = [(function, data, i, pca_dims, splits) for i in tests]
    checkpoint_file = f'{checkpoint_dir}/{model}.csv'
    last = 0
    if os.path.exists(checkpoint_file):
        last = pd.read_csv(checkpoint_file).shape[0]
        print(f'Last detected point is {last}, skipping...')
    ranges = [(i, i + tests_checkpoint if i + tests_checkpoint < len(tests) else len(tests)) for i in
              range(last, len(tests), tests_checkpoint)]
    for checkpoint_start, checkpoint_end in ranges:
        print(f'Checkpoint {checkpoint_start}-{checkpoint_end}')
        try:
            with mp.Pool(processes=mp.cpu_count()//2, initializer=worker_init, initargs=(42,)) as pool:
                results = [i for i in pool.map(util.cross_validate, tests[checkpoint_start:checkpoint_end])]
                results = pd.DataFrame(results)
                if os.path.exists(checkpoint_file):
                    checkpoint_results = pd.read_csv(checkpoint_file, index_col=False)
                    results = pd.concat([checkpoint_results, results], ignore_index=True)
                results.to_csv(checkpoint_file, index=False)
        finally:
            pool.close()
            pool.join()
    checkpoint_data = pd.read_csv(checkpoint_file)
    if checkpoint_data.shape[0] == len(tests):
        checkpoint_data.to_csv(path)
        os.remove(checkpoint_file)
    else:
        print(f'Something wnet wrong, checkpoint has {checkpoint_data.shape[0]} records of {len(tests)}')


# if __name__ == "__main__":
#     try:
#         mp.set_start_method('spawn')
#     finally:
#         print('tests start')
#     tests = combine_tests('DecisionTreeClassifier')
#     x, y = dm.load('../data/in_csv/x20', '../data/in_csv/y')
#     x_train, y_train, x_test, y_test = util.train_test_from(x, y)
#     # conduct_tests('DecisionTreeClassifier', (x_train, y_train, x_test, y_test), '../results/DTtest.csv',
#     #               tests=combine_tests('DecisionTreeClassifier')[:mp.cpu_count()])
#     # conduct_tests('ExtraTreesClassifier', (x_train, y_train, x_test, y_test), '../results/ETtest.csv',
#     #               tests=combine_tests('ExtraTreesClassifier')[:mp.cpu_count()])
#     # conduct_tests('RandomForestClassifier', (x_train, y_train, x_test, y_test), '../results/RFtest.csv',
#     #               tests=combine_tests('RandomForestClassifier')[:mp.cpu_count()])
#     # conduct_tests('XGBClassifier', (x_train, y_train, x_test, y_test), '../results/XGBtest.csv',
#     #               tests=combine_tests('XGBClassifier')[:mp.cpu_count()])
#     conduct_tests('DecisionTreeClassifier', (x_train, y_train, x_test, y_test), '../results/DTtest.csv')
#     conduct_tests('ExtraTreesClassifier', (x_train, y_train, x_test, y_test), '../results/ETtest.csv')
#     conduct_tests('RandomForestClassifier', (x_train, y_train, x_test, y_test), '../results/RFtest.csv')
#     conduct_tests('XGBClassifier', (x_train, y_train, x_test, y_test), '../results/XGBtest.csv')

if len(sys.argv) > 1 and sys.argv[1] == "__bilinear__":
    sys.argv[1] = "consumed"
    for i in range(20, 1, -1):
        x, y = dm.load(f'../data/in_csv/x{i}', '../data/in_csv/y')
        x_train, y_train, x_test, y_test = util.train_test_from(x, y)
        print('DecisionTreeClassifier')
        if os.path.exists(f'../results/bilDT/DT{i}.csv'):
            print('skip DecisionTreeClassifier')
        else:
            conduct_tests('DecisionTreeClassifier', (x_train, y_train, x_test, y_test), f'../results/bilDT/DT{i}.csv')
        print('ExtraTreesClassifier')
        if os.path.exists(f'../results/bilET/ET{i}.csv'):
            print('skip ExtraTreesClassifier')
        else:
            conduct_tests('ExtraTreesClassifier', (x_train, y_train, x_test, y_test), f'../results/bilET/ET{i}.csv')
        print('RandomForestClassifier')
        if os.path.exists(f'../results/bilRF/RF{i}.csv'):
            print('skip RandomForestClassifier')
        else:
            conduct_tests('RandomForestClassifier', (x_train, y_train, x_test, y_test), f'../results/bilRF/RF{i}.csv')
        print('XGBClassifier')
        if os.path.exists(f'../results/bilXGB/XGB{i}.csv'):
            print('skip XGBClassifier')
        else:
            conduct_tests('XGBClassifier', (x_train, y_train, x_test, y_test), f'../results/bilXGB/XGB{i}.csv')

if len(sys.argv) > 1 and sys.argv[1] == "__pca__":
    sys.argv[1] = "consumed"
    x, y = dm.load(f'../data/in_csv/x2', '../data/in_csv/y')
    x_train, y_train, x_test, y_test = util.train_test_from(x, y)
    for i in range(60, 19, -5):
        print(f'pca{i}')
        x_train, x_test, variance = util.use_pca(x_train, x_test, i)
        print('DecisionTreeClassifier')
        if os.path.exists(f'../results/pcaDT/DT{i}.csv'):
            print('skip DecisionTreeClassifier')
        else:
            conduct_tests('DecisionTreeClassifier', (x_train, y_train, x_test, y_test), f'../results/pcaDT/DT{i}.csv')
        print('ExtraTreesClassifier')
        if os.path.exists(f'../results/pcaET/ET{i}.csv'):
            print('skip ExtraTreesClassifier')
        else:
            conduct_tests('ExtraTreesClassifier', (x_train, y_train, x_test, y_test), f'../results/pcaET/ET{i}.csv')
        print('RandomForestClassifier')
        if os.path.exists(f'../results/pcaRF/RF{i}.csv'):
            print('skip RandomForestClassifier')
        else:
            conduct_tests('RandomForestClassifier', (x_train, y_train, x_test, y_test), f'../results/pcaRF/RF{i}.csv')
        print('XGBClassifier')
        if os.path.exists(f'../results/pcaXGB/XGB{i}.csv'):
            print('skip XGBClassifier')
        else:
            conduct_tests('XGBClassifier', (x_train, y_train, x_test, y_test), f'../results/pcaXGB/XGB{i}.csv')

if len(sys.argv) > 1 and sys.argv[1] == "__cv-bil__":
    sys.argv[1] = "consumed"
    for name in ["DT", "ET", "RF", "XGB"]:
        if not os.path.exists(f'../results/cv-bil{name}'):
            os.mkdir(f'../results/cv-bil{name}')
    for i in range(20, 1, -1):
        print(f'cv-bil{i}')
        for fullname, shortname in [('DecisionTreeClassifier', 'DT'), ('ExtraTreesClassifier', 'ET'),
                                    ('RandomForestClassifier', 'RF'), ('XGBClassifier', 'XGB')]:
            if os.path.exists(f'../results/cv-bil{shortname}/{shortname}{i}.csv'):
                print(f'skip {fullname}')
            else:
                x, y = dm.load(f'../data/in_csv/x{i}', '../data/in_csv/y')
                df = pd.read_csv(f'../results/bil{shortname}/{shortname}{i}.csv').sort_values('accuracy',
                                                                                              ascending=False)
                accuracy_threshold = df['accuracy'].quantile(0.95)
                df = df[df['accuracy'] >= accuracy_threshold]
                tests = util.df_to_param_dicts(df)
                conduct_tests_cv(fullname, (x, y), f'../results/cv-bil{shortname}/{shortname}{i}.csv', tests=tests)

if len(sys.argv) > 1 and sys.argv[1] == "__cv-pca__":
    sys.argv[1] = "consumed"
    for name in ["DT", "ET", "RF", "XGB"]:
        if not os.path.exists(f'../results/cv-pca{name}'):
            os.mkdir(f'../results/cv-pca{name}')
    x, y = dm.load('../data/extended_csv/x2', '../data/extended_csv/y')
    for i in range(60, 19, -5):
        print(f'pca{i}')
        for fullname, shortname in [('DecisionTreeClassifier', 'DT'), ('ExtraTreesClassifier', 'ET'),
                                    ('RandomForestClassifier', 'RF'), ('XGBClassifier', 'XGB')]:
            if os.path.exists(f'../results/cv-pca{shortname}/{shortname}{i}.csv'):
                print(f'skip {fullname}')
            else:
                df = pd.read_csv(f'../results/pca{shortname}/{shortname}{i}.csv').sort_values('accuracy',
                                                                                              ascending=False)
                accuracy_threshold = df['accuracy'].quantile(0.95)
                df = df[df['accuracy'] >= accuracy_threshold]
                tests = util.df_to_param_dicts(df)
                conduct_tests_cv(fullname, (x, y), f'../results/cv-pca{shortname}/{shortname}{i}.csv', tests=tests,
                                 pca_dims=i)
