import itertools
import math
import re

import util
import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kruskal

metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'time']
colors = ['blue', 'red', 'green', 'orange', 'purple', 'lime', 'black', 'yellow', 'cyan', 'magenta', 'brown', 'pink',
          'teal', 'gray']
lower_means_better = {'time'}
excluded = {'mae', 'loss'}


def save_plot(path):
    if not os.path.exists(path):
        plt.savefig(path)
        plt.clf()


def drop_unnamed(df):
    redundant = df.columns[0]
    df = df.drop(redundant, axis=1)
    return df


def values_for_columns(df):
    values_dict = dict()
    for column in set(df.columns).difference(set(metrics_names)):
        values = set()
        for _, row in df.iterrows():
            values.add(row[column])
        values_dict[column] = values
    return values_dict


# ---

def visualise_hiperparameters_influence(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.fillna('None')
    df = drop_unnamed(df)
    values_dict = values_for_columns(df)
    fig, axes = plt.subplots(len(metrics_names), len(values_dict.keys()),
                             figsize=(5 * len(values_dict.keys()), (len(metrics_names) + 1) * 5))
    for m in range(len(metrics_names)):
        metric = metrics_names[m]
        iterator = -1
        for param, param_values in values_dict.items():
            iterator += 1
            for p in range(len(param_values)):
                p_val = list(param_values)[p]
                m_vals = [row[metric] for _, row in df.iterrows() if row[param] == p_val]
                m_vals = pd.Series(pd.cut(m_vals, bins=10)).value_counts(sort=False)
                m_vals.plot(kind='bar', ax=axes[m, iterator], position=p, width=0.1, color=colors[p],
                            label=str(p_val))
                axes[m, iterator].set_xlabel(str(metric))
                axes[m, iterator].set_ylabel('counts')
                axes[m, iterator].set_title(str(param))
                axes[m, iterator].legend()
    plt.tight_layout()
    util.save_plot(output_path)


def visualise_metrics_correlations(intput_dir, output_path):
    files = [i for i in os.listdir(intput_dir) if i.endswith('.csv')]
    data = {util.capital_letters(i): drop_unnamed(pd.read_csv(f'{intput_dir}/{i}').fillna('None')) for i in files}
    correlasions = list(itertools.combinations(metrics_names, 2))
    horizontal, vertical = util.best_rect_dims_of_(len(correlasions))
    fig, axes = plt.subplots(vertical, horizontal,
                             figsize=(5 * horizontal, 5 * vertical))
    for h in range(horizontal):
        for v in range(vertical):
            horizontal_metric, vertical_metric = correlasions[v * horizontal + h]
            for name, results in data.items():
                pairs = [(row[horizontal_metric], row[vertical_metric]) for _, row in results.iterrows()]
                h_vals, v_vals = [i[0] for i in pairs], [i[1] for i in pairs]
                axes[v, h].scatter(h_vals, v_vals, alpha=0.2, label=name)
            axes[v, h].set_xlabel(horizontal_metric)
            axes[v, h].set_ylabel(vertical_metric)
            axes[v, h].set_title(f'{horizontal_metric} to {vertical_metric}')
            axes[v, h].legend()
    plt.tight_layout()
    util.save_plot(output_path)


# ---

def visualise_all_results_over_files(input_dir, output_path):
    files = [i for i in os.listdir(input_dir) if i.endswith('.csv')]
    if len(files) == 0:
        return
    files = sorted(files, key=lambda x: util.int_from_str(x))
    x_vals = [util.int_from_str(file) for file in files]
    horizontal, vertical = (3, 2)
    fig, axes = plt.subplots(vertical, horizontal,
                             figsize=(5 * horizontal, 5 * vertical))
    for h in range(horizontal):
        for v in range(vertical):
            metric = metrics_names[h * vertical + v]
            y_vals = [pd.read_csv(f'{input_dir}/{file}')[metric] for file in files]
            axes[v, h].boxplot(y_vals, labels=x_vals)
            axes[v, h].set_title(metric)
    plt.tight_layout()
    save_plot(output_path)


def visualise_best_results_over_files(input_dir, output_path):
    files = [i for i in os.listdir(input_dir) if i.endswith('.csv')]
    if len(files) == 0:
        return
    files = sorted(files, key=lambda x: util.int_from_str(x))
    x_vals = [util.int_from_str(file) for file in files]
    horizontal, vertical = (3, 2)
    fig, axes = plt.subplots(vertical, horizontal,
                             figsize=(5 * horizontal, 5 * vertical))
    for h in range(horizontal):
        for v in range(vertical):
            metric = metrics_names[h * vertical + v]
            y_vals = [pd.read_csv(f'{input_dir}/{file}')[metric] for file in files]
            ascending = metric in lower_means_better
            best = [val.min() if ascending else val.max() for val in y_vals]
            avg_top10 = [val.sort_values(ascending=ascending)[:10].mean() for val in y_vals]
            avg_quantile95 = [val[val < val.quantile(0.95) if ascending else val > val.quantile(0.95)].mean() for val in
                              y_vals]
            axes[v, h].plot(x_vals, best, color=colors[0], label='best')
            axes[v, h].plot(x_vals, avg_top10, color=colors[1], label='avg of top 10')
            axes[v, h].plot(x_vals, avg_quantile95, color=colors[2], label='avg of top 5%')
            axes[v, h].set_title(metric)
            axes[v, h].legend()
    plt.tight_layout()
    save_plot(output_path)


def visualise_hiperparemetrs_influence_over_files(input_dir, output_path):
    files = [i for i in os.listdir(input_dir) if i.endswith('.csv')]
    if len(files) == 0:
        return
    files = sorted(files, key=lambda x: util.int_from_str(x))
    x_vals = [util.int_from_str(file) for file in files]
    horizontal, vertical = (3, 2)
    fig, axes = plt.subplots(vertical, horizontal,
                             figsize=(5 * horizontal, 5 * vertical))
    data = [drop_unnamed(pd.read_csv(f'{input_dir}/{file}')) for file in files]
    cols = list(set(data[0].columns).difference(metrics_names).difference(excluded))
    for h in range(horizontal):
        for v in range(vertical):
            metric = metrics_names[h * vertical + v]
            for i in range(len(cols)):
                col_name = cols[i]
                groups = [[df[df[col_name] == val][metric] for val in df[col_name].unique()] for df in data]
                group_sizes = [len(g) for g in groups]
                data_sizes = [len(df) for df in data]
                kruskals = [kruskal(*g) for g in groups]
                etas = [(kruskals[i][0], kruskals[i][1], group_sizes[i], data_sizes[i]) for i in range(len(groups))]
                etas = [float(e[0] - e[2] + 1) / float(e[3] - e[2]) for e in etas]

                axes[v, h].plot(x_vals, etas, color=colors[i], label=col_name)
            axes[v, h].set_title(metric)
            axes[v, h].legend()
    plt.tight_layout()
    save_plot(output_path)


graphs_dir = '../graphs'
for name in os.listdir('../results'):
    input_dir = f'../results/{name}'
    if os.path.isdir(input_dir):
        visualise_all_results_over_files(input_dir, f'{graphs_dir}/all/all_{name}')
        visualise_best_results_over_files(input_dir, f'{graphs_dir}/best/best_{name}')
        visualise_hiperparemetrs_influence_over_files(input_dir, f'{graphs_dir}/kruskal/kruskal_{name}')
