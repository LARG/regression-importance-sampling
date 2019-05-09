"""Plot singlepath results."""
from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import results_pb2

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', help='Result directory with results to load.')
parser.add_argument('--plot_file', help='File to save plot to.')
FLAGS = parser.parse_args()

# The confidence level. 1.96 is for 95% confidence intervals
Z_SCORE = 1.96

# Plot params
plot_params = {'bfont': 30,
               'lfont': 30,
               'tfont': 25,
               'legend': True,
               'legend_loc': 1,
               'legend_cols': 2,
               'y_range': [10e-8, 1],
               'x_range': [100, 1e4],
               'y_range': None,
               'x_range': None,
               'log_scale': True,
               'x_label': 'Number of Trajectories',
               'y_label': 'Mean Squared Error',
               'plot_error': True,
               'shade_error': True,
               'x_mult': 100}

methods_to_plot = ['RIS(1)', 'RIS(4)', 'RIS(5)', 'IS', 'REG', 'OIS']


class LineStyle(object):  # noqa
    def __init__(self, style, color, width, marker=None, markersize=None, dashes=None, alpha=0.5):  # noqa
        self.color = color
        self.style = style
        self.width = width
        self.marker = marker
        self.markersize = markersize
        self.dashes = dashes
        self.alpha = alpha


def get_line_style(label):  # noqa

    if label == 'RIS(0)':
        return LineStyle('-', 'b', 3, alpha=0.25)
    elif label == 'OIS':
        return LineStyle('--', 'r', 3, alpha=0.25)
    elif label == 'REG':
        return LineStyle('-', 'g', 6, alpha=0.25)
    elif label == 'RIS(2)':
        return LineStyle('-', 'k', 3, alpha=0.25)
    elif label == 'RIS(4)':
        return LineStyle('-.', 'k', 3, alpha=0.25)
    elif label == 'RIS(1)':
        return LineStyle('--', 'g', 3, alpha=0.25)
    elif label == 'RIS(3)':
        return LineStyle(':', 'c', 4, alpha=0.25)
    else:
        return None


def set_line_style(label, line):  # noqa
    style = get_line_style(label)
    if style is None:
        return
    line.set_linestyle(style.style)
    if style.marker is not None:
        line.set_marker(style.marker)
    if style.markersize is not None:
        line.set_markersize(style.markersize)
    line.set_color(style.color)
    line.set_linewidth(style.width)
    if style.dashes is not None:
        line.set_dashes(style.dashes)


def get_label(label):  # noqa

    strip_len = len('mse') + 1
    label = label[:-strip_len]
    if label == 'IS':
        return 'OIS'
    return label


def load_results(result_directory):  # noqa

    data = {}
    results = results_pb2.Results()
    for filename in os.listdir(result_directory):
        with open(os.path.join(result_directory, filename), 'rb') as f:
            try:
                results.ParseFromString(f.read())
            except Exception as e:
                # Checking if is cluster stderr file.
                if filename.endswith('.err'):
                    continue
                raise e
            for method in results.methods:
                mse_key = '%s_mse' % method.method_name
                if mse_key not in data:
                    data[mse_key] = []
                data[mse_key].append(np.array(method.mse))

    return data


def compute_stats(data):  # noqa
    means = {}
    confidences = {}
    zero_conf = False
    for key in data:
        data[key] = np.array(data[key])

        means[key] = np.mean(data[key], axis=0)
        confidences[key] = (Z_SCORE * np.std(data[key], axis=0) /
                            np.sqrt(np.size(data[key], axis=0)))
        print(key, np.size(data[key], axis=0))
        if zero_conf:
            confidences[key] = np.zeros(np.size(means[key]))
    return means, confidences


def main():  # noqa

    if not FLAGS.result_directory:
        print('Must provide results directory.')
        return

    data = load_results(FLAGS.result_directory)
    means, confidences = compute_stats(data)

    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)

    ktp = ('%s_mse' % key for key in methods_to_plot if '%s_mse' % key in means)
    for key in ktp:

        y_data = means[key]
        err = confidences[key]
        label = get_label(key)
        print('Checking', label)
        if methods_to_plot is not None and label not in methods_to_plot:
            continue

        if 'inf' in label:
            label = 'RIS(5)'
        if label.startswith('RIS'):
            print (label)
            end = label.find(')')
            num = int(label[4:end]) - 1
            label = '%s%d%s' % (label[:4], num, label[end:])

        style = get_line_style(label)
        xs = np.arange(len(y_data)) + 1

        # To account for the fact that the first 100 points are sampled more frequently.
        # See singlepath.py and how eval_freq is used.
        xs *= plot_params['x_mult']

        line, = plt.plot(xs, y_data, label=label)
        alpha = 0.25 if style is None else style.alpha
        color = line.get_color() if style is None else style.color
        plt.fill_between(xs, y_data - err, y_data + err, alpha=alpha, facecolor=color)

        if style is not None:
            set_line_style(label, line)

    x_title = plot_params['x_label']
    y_title = plot_params['y_label']
    if plot_params['log_scale']:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if plot_params['y_range'] is not None:
        plt.ylim(plot_params['y_range'])
    if plot_params['x_range'] is not None:
        plt.xlim(plot_params['x_range'])
    ax.set_xlabel(x_title, fontsize=plot_params['bfont'])
    ax.set_ylabel(y_title, fontsize=plot_params['bfont'])
    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])

    if FLAGS.plot_file is None:
        plt.show()
    else:
        fig.savefig(FLAGS.plot_file)


if __name__ == '__main__':
    main()
