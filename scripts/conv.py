import random
import sys

from tensorflow.keras.optimizers import RMSprop
import multiprocessing as mp
import util
import itertools
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import pandas as pd
import numpy as np
import os
import cv2
from data_manipulation import slice_image, shift_to_center, symbols_names, minimise

pos_filters = [[32, 64, 128], [16, 32, 64], [32, 64], [64, 128]]
pos_dense = [[256, 128, 64], [128, 64, 32], [256, 128], [128, 64]]
pos_dropout = [0.3, 0.2]
pos_kernel = [3, 4, 5]
pos_epochs = [20, 40, 60]
pos_learning_rates = [2e-4, 5e-4, 2e-3, 5e-3]


def combine_tests():
    tests = [list(i) for i in
             itertools.product(pos_filters, pos_dense, pos_dropout, pos_kernel, pos_epochs,
                               pos_learning_rates)]
    tests = [{'filters': i[0], 'dense': i[1], 'dropout': i[2], 'kernel': i[3], 'epochs': i[4],
              'learning_rate': i[5]} for i in tests]
    return tests


# ----
def load_images(input_dir, scale=20):
    all_images = []
    labels = []
    for path in [f'{input_dir}/{file}' for file in os.listdir(input_dir)]:
        img = cv2.imread(path)
        sliced = slice_image(img)
        sliced = [shift_to_center(i) for i in sliced]
        all_images.extend(sliced)
        labels.extend([i // len(symbols_names) for i in range(len(sliced))])
    all_images = [minimise(i, scale) for i in all_images]
    all_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in all_images]
    all_images = np.array(all_images, dtype=np.float32) / 255.0
    return all_images, labels


# ----

def build_model(filters=[32, 64, 128], kernel=3, pooling=2, dropout=0.3, dense=[256, 128, 64]):
    model = Sequential([Input(shape=(20, 20, 1))])
    for f in filters:
        model.add(
            Conv2D(f, kernel_size=(kernel, kernel), padding='same', activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(
            Conv2D(f, kernel_size=(kernel, kernel), padding='same', activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pooling, pooling)))
        model.add(Dropout(dropout, seed=42))
    model.add(Flatten())
    for d in dense:
        model.add(Dense(d, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(dropout, seed=42))
    model.add(Dense(12, activation='softmax'))
    return model


def compile_net(model, optimiser, learning_rate):
    model.compile(optimizer=optimiser(learning_rate=learning_rate),
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
    return model


def fit_evaluate(model, data, epochs, batch_size):
    x_train, y_train, x_test, y_test = data
    start = util.measure_time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    end = util.measure_time(start)
    probabilities = model.predict(x_test)
    return util.evaluate(probabilities, y_test, training_time=end)


# ----

def use_net(args):
    data, test = args
    model = build_model(filters=test['filters'], dense=test['dense'], kernel=test['kernel'], dropout=test['dropout'])
    model = compile_net(model, RMSprop, test['learning_rate'])
    result = fit_evaluate(model, data, test['epochs'], 32)
    for k, v in test.items():
        result[k] = v
    return result


def worker_init(seed):
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    random.seed(seed)


def conduct_tests(data, path, tests=None, tests_checkpoint=25, checkpoint_file='../tmp/checkpoint.csv'):
    worker_init(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
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
        print(f'Checkpoint {checkpoint_start}-{checkpoint_end} of {len(tests)}')
        try:
            with mp.Pool(processes=mp.cpu_count(), initializer=worker_init, initargs=(42,)) as pool:
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


if len(sys.argv) > 1 and sys.argv[1] == "__conv__":
    sys.argv[1] = "consumed"
    if os.path.exists('../results/CONVTest.csv'):
        sys.exit(0)
    mp.set_start_method("spawn", force=True)
    x, y = load_images('../data/extended')
    x_train, y_train, x_test, y_test = util.train_test_from(x, y)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    data = (x_train, y_train, x_test, y_test)
    tests = combine_tests()
    conduct_tests(data, '../results/CONVTest.csv', tests=tests)
