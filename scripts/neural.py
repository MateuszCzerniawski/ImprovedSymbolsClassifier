import itertools
import os
import random
import sys
import warnings

import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Sequential
import tensorflow
import multiprocessing as mp
import data_manipulation as dm
import util

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pos_epochs = [50, 75, 100]
pos_batches = [32, 64, 128]
pos_unit_counts = [256, 128, 64, 32]
pos_reg_vals = [2e-4, 5e-4, 2e-3]
pos_learning_rates = [2e-4, 5e-4, 2e-3, 5e-3]
pos_optimizers = [Adam, RMSprop]
pos_regularizes = ['no_reg', 'l1', 'l2']


def build_net(inputs, output, units_counts, function='elu', regularizer='no_reg', regularization=None):
    regularizer = {'no_reg': 'no_reg', 'l1': l1, 'l2': l2}[regularizer]
    if regularizer == 'no_reg':
        regularization = None
    model = Sequential()
    model.add(keras.Input(shape=(inputs,)))
    for units in units_counts:
        reg = regularizer(regularization) if regularizer != 'no_reg' else None
        model.add(Dense(units, activation=function, kernel_regularizer=reg))
    model.add(Dense(output, activation='softmax'))
    return model


def compile_net(model, optimiser, learning_rate):
    model.compile(optimizer=optimiser(learning_rate=learning_rate),
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def fit_evaluate(model, data, epochs, batch_size):
    x_train, y_train, x_test, y_test = data
    start = util.measure_time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    end = util.measure_time(start)
    probabilities = model.predict(x_test)
    return util.evaluate(probabilities, y_test, training_time=end)


def combine_tests(reduced=False):
    units = []
    if not reduced:
        for i in range(1, len(pos_unit_counts) + 1):
            units.extend([list(i) for i in itertools.combinations(pos_unit_counts, i)])
    else:
        units = [[256, 128], [256, 128, 64], [256, 64, 32], [256, 32, 16]]
    regs = [list(i) for i in itertools.product(pos_regularizes[1:], pos_reg_vals)]
    regs.append([pos_regularizes[0], 0])
    tests = [list(i) for i in
             itertools.product(units, pos_epochs, pos_batches, pos_learning_rates, pos_optimizers, regs)]
    tests = [{'units': i[0], 'epochs': i[1], 'batch_size': i[2], 'learning_rate': i[3],
              'optimiser': i[4], 'regularizer': i[5][0], 'reg_val': i[5][1]} for i in tests]
    return tests


def use_net(data):
    (x_train, y_train, x_test, y_test), params = data
    net = build_net(x_train.shape[1], len(dm.symbols_names), params['units'],
                    regularizer=params['regularizer'], regularization=params['reg_val'])
    net = compile_net(net, params['optimiser'], params['learning_rate'])
    results = fit_evaluate(net, (x_train, y_train, x_test, y_test), params['epochs'], params['batch_size'])
    for key, value in results.items():
        params[key] = value
    return params


def conduct_tests(data, path, tests=None, tests_checkpoint=25, checkpoint_file='../tmp/checkpoint.csv'):
    if tests is None:
        tests = combine_tests()
    tests = [(data, i) for i in tests]
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
                results = [i for i in pool.map(use_net, tests[checkpoint_start:checkpoint_end])]
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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    random.seed(seed)


def conduct_tests_cv(data, path, tests, pca_dims=None, splits=5, tests_checkpoint=25,
                     checkpoint_file='../tmp/checkpoint.csv'):
    worker_init(42)
    tests = [(use_net, data, i, pca_dims, splits) for i in tests]
    last = 0
    if os.path.exists(checkpoint_file):
        last = pd.read_csv(checkpoint_file).shape[0]
        print(f'Last detected point is {last}, skipping...')
    ranges = [(i, i + tests_checkpoint if i + tests_checkpoint < len(tests) else len(tests)) for i in
              range(last, len(tests), tests_checkpoint)]
    for checkpoint_start, checkpoint_end in ranges:
        print(f'Checkpoint {checkpoint_start}-{checkpoint_end} of {len(tests)}')
        try:
            with mp.Pool(processes=mp.cpu_count() // 2, initializer=worker_init, initargs=(42,)) as pool:
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
#     print('tu')
#     x, y = dm.load('../data/in_csv/x2', '../data/in_csv/y')
#     x_train, y_train, x_test, y_test = util.train_test_from(x, y)
#     x_train, x_test, variance = util.use_pca(x_train, x_test, 55, pca_export_path='../results/pca.joblib')
#     tests = combine_tests(reduced=True)
#     print(len(tests))
#     conduct_tests((x_train, y_train, x_test, y_test), '../results/NETStest.csv', tests=tests)

if len(sys.argv) > 1 and sys.argv[1] == "__bilinear__":
    sys.argv[1] = "consumed"
    tests = combine_tests()
    tests = [t for t in tests if len(t['units']) in {2, 3}]
    print(f'{len(tests)} to go...')
    for i in range(20, 1, -1):
        print(f'bil{i}')
        path = f'../results/bilNETS/NETS{i}.csv'
        if os.path.exists(path):
            print('skip')
            continue
        x, y = dm.load(f'../data/in_csv/x{i}', '../data/in_csv/y')
        x_train, y_train, x_test, y_test = util.train_test_from(x, y)
        conduct_tests((x_train, y_train, x_test, y_test), path, tests=tests)

if len(sys.argv) > 1 and sys.argv[1] == "__pca__":
    sys.argv[1] = "consumed"
    x, y = dm.load('../data/in_csv/x2', '../data/in_csv/y')
    x_train, y_train, x_test, y_test = util.train_test_from(x, y)
    tests = combine_tests()
    tests = [t for t in tests if len(t['units']) in {2, 3}]
    for i in range(60, 19, -5):
        print(f'pca{i}')
        path = f'../results/pcaNETS/NETS{i}.csv'
        if os.path.exists(path):
            print('skip')
            continue
        x_train, x_test, variance = util.use_pca(x_train, x_test, i)
        conduct_tests((x_train, y_train, x_test, y_test), path, tests=tests)

if len(sys.argv) > 1 and sys.argv[1] == "__cv-bil__":
    sys.argv[1] = "consumed"
    if not os.path.exists('../results/cv-bilNETS'):
        os.mkdir('../results/cv-bilNETS')
    for i in range(20, 1, -1):
        print(f'cv-bil{i}')
        if os.path.exists(f'../results/cv-bilNETS/NETS{i}.csv'):
            print('skip')
            continue
        x, y = dm.load(f'../data/extended_csv/x{i}', '../data/extended_csv/y')
        df = pd.read_csv(f'../results/bilNETS/NETS{i}.csv').sort_values('accuracy', ascending=False)
        accuracy_threshold = df['accuracy'].quantile(0.95)
        df = df[df['accuracy'] >= accuracy_threshold]
        tests = util.df_to_param_dicts(df)
        conduct_tests_cv((x, y), f'../results/cv-bilNETS/NETS{i}.csv', tests)

if len(sys.argv) > 1 and sys.argv[1] == "__cv-pca__":
    sys.argv[1] = "consumed"
    x, y = dm.load('../data/extended_csv/x2', '../data/extended_csv/y')
    if not os.path.exists('../results/cv-pcaNETS'):
        os.mkdir('../results/cv-pcaNETS')
    for i in range(60, 19, -5):
        print(f'pca{i}')
        if os.path.exists(f'../results/cv-pcaNETS/NETS{i}.csv'):
            print('skip')
            continue
        df = pd.read_csv(f'../results/pcaNETS/NETS{i}.csv').sort_values('accuracy', ascending=False)
        accuracy_threshold = df['accuracy'].quantile(0.95)
        df = df[df['accuracy'] >= accuracy_threshold]
        tests = util.df_to_param_dicts(df)
        conduct_tests_cv((x, y), f'../results/cv-pcaNETS/NETS{i}.csv', tests, pca_dims=i)
