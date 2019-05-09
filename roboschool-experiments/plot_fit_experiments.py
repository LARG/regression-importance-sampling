"""Plot curves for MSE as fitting policy."""
import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from protos import blackbox_results_pb2


parser = argparse.ArgumentParser()
parser.add_argument('result_directory', default=None, help='Directory with results files.')
parser.add_argument('--plot_file', default=None, help='File to write plot to.')
FLAGS = parser.parse_args()

rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

hopper = True
hopper = False

plot_params = {'bfont': 28,
               'lfont': 28,
               'tfont': 25,
               'legend': False,
               'legend_loc1': 0,
               'legend_loc2': 4,
               'legend_loc3': 0,
               'legend_cols': 3,
               'y1_range': [0, 700],  # Cheetah, Hopper
               'y2_range': [5.79, 6.2],
               'y3_range': [0, 150],
               'x_range': [0, 1000],
               'x_label': 'Training Step',
               'y1_label': 'MSE',
               'y2_label': 'Neg. Log Likelihood',
               'y3_label': 'Value',
               'x_mult': 25,
               'plot_min': True}

Z_SCORE = 1.96
# Best fit MSE values taken from plot_bar_graph.py output for Cheetah
flat_y = 581.2067376708984
flat_std = 103.67551291055993
flat_n = 200

# Toggle different lines on/off for presentation figures.
plot_ois = True
plot_ris = True
plot_flat = False
plot_loss = True
loss_blank = False

methods_to_plot = ['test_1_1_64']

if True in [plot_ois, plot_ris, plot_flat, plot_loss]:
    plot_params['legend'] = True
    if plot_loss and loss_blank:
        plot_params['legend'] = False

# If plotting hopper results then use these numbers.
if hopper:
    plot_params['y1_range'] = [0, 700]
    plot_params['y2_range'] = [2.25, 2.35]
    plot_params['y3_range'] = [0, 300]
    plot_params['x_range'] = [0, 7000]
    plot_params['x_mult'] = 100
    # Best fit MSE values taken from plot_bar_graph.py output for Hopper
    flat_y = 339.3650066884607
    flat_std = 242.3488541185939
    flat_n = 200


class LineStyle(object):

    def __init__(self, style, color, width, marker=None, markersize=None,
                 dashes=None, alpha=0.5):
        self.color = color
        self.style = style
        self.width = width
        self.marker = marker
        self.markersize = markersize
        self.dashes = dashes
        self.alpha = alpha


def get_line_style(label):
    lower = label.lower()
    thickness = 4
    if 'ris' in lower:
        return LineStyle('-', 'b', thickness, alpha=0.5)
    elif 'ois' in lower:
        return LineStyle('--', 'g', thickness)
    else:
        return LineStyle('-', 'k', 1)


def set_line_style(label, line, eb_h=None, eb_v=None, itr=None):
    style = get_line_style(label)
    line.set_linestyle(style.style)
    if style.marker is not None:
        line.set_marker(style.marker)
    if style.markersize is not None:
        line.set_markersize(style.markersize)
    line.set_color(style.color)
    line.set_linewidth(style.width)
    if style.dashes is not None:
        line.set_dashes(style.dashes)
    if eb_h is not None:
        for cap in eb_h:
            cap.set_color(style.color)
            cap.set_markeredgewidth(style.width)
    if eb_v is not None:
        eb_v[0].set_linewidth(style.width)
        eb_v[0].set_color(style.color)
    if itr is not None and itr % 2 != 0:
        line.set_linestyle('--')


def read_proto(filename):
    results = blackbox_results_pb2.FitResults()
    try:
        with open(filename, 'rb') as f:
            results.ParseFromString(f.read())
    except:
        return None
    return results


def get_label(name):
    lower = name.lower()
    if 'proportion' in lower or 'reg' in lower:
        return 'Regression'
    elif 'density' in lower or 'is' in lower:
        return 'Density'
    elif 'model' in lower:
        return 'Model'
    else:
        return name


def main():

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = {}
    est_data = {}
    metrics = {'train_loss': {}, 'validation_loss': {}}
    true_IS_mse, true_IS_estimate = [], []

    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        results = read_proto(filename)
        if results is None:
            continue
        method = '_'.join(basename.split('_')[:-1])
        if method not in methods_to_plot:
            continue

        if method not in data:
            data[method] = []
            est_data[method] = []
            for metric in metrics:
                if method not in metrics[metric]:
                    metrics[metric][method] = []
        if len(results.mse) == 0:
            continue
        data[method].append(np.array(results.mse))
        est_data[method].append(np.array(results.estimates))
        metrics['train_loss'][method].append(np.array(results.losses[1:]))
        metrics['validation_loss'][method].append(
            np.array(results.validation_loss))
        true_IS_mse.append(results.density_mse)
        true_IS_estimate.append(results.density_estimate)

    print('MSE', np.mean(true_IS_mse))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    fig.set_size_inches(13.5, 12.0, forward=True)

    length = 0
    min_v_loss_point = 0
    lines1, lines2, lines3 = [], [], []
    labels1, labels2, labels3 = [], [], []
    styles = ['-', '--', '-.', ':']

    def plot_method(ax, data, method):
        mean = np.mean(np.array(data[method]), axis=0)
        std = np.std(np.array(data[method]), axis=0)
        n = np.size(np.array(data[method]), axis=0)
        print(method, n)
        x = np.arange(len(mean)) * plot_params['x_mult']
        yerr = Z_SCORE * std / np.sqrt(float(n))
        print('n', np.sqrt(float(n)))
        name = 'RIS'

        if plot_ris:
            style = get_line_style(name)
            color, alpha = style.color, style.alpha
            line, = ax.plot(x, mean, color=color)
            set_line_style(name, line)
            ax.fill_between(x, mean - yerr, mean + yerr, facecolor=color, alpha=alpha)
            return line, name

    for i, method in enumerate(data):
        if methods_to_plot is not None and method not in methods_to_plot:
            continue
        line, name = plot_method(ax2, data, method)
        lines1.append(line)
        labels1.append(name)
        length = len(data[method][0])

        line, name = plot_method(ax3, est_data, method)
        lines3.append(line)
        labels3.append(name)

        if plot_loss:
            data_list = [metrics[m][method] for m in metrics]
            labels = ['Validation', 'Train Loss']
            if loss_blank:
                labels = []
            cs = ['k', 'r']
            styles = ['-.', ':']
            valid_mean = np.mean(np.array(data_list[0]), axis=0)
            x = np.argmin(valid_mean)
            min_v_loss_point = x * plot_params['x_mult']

            if not loss_blank:
                for j, data in enumerate(data_list):

                    mean = np.mean(np.array(data), axis=0)
                    std = np.std(np.array(data), axis=0)
                    n = np.size(np.array(data), axis=0)
                    yerr = Z_SCORE * std / np.sqrt(float(n))
                    x = np.arange(len(mean)) * plot_params['x_mult']
                    line, = ax1.plot(x, mean, c=cs[j], linestyle=styles[j])
                    line.set_linewidth(2)
                    ax1.fill_between(x, mean - yerr, mean + yerr, facecolor=cs[j], alpha=0.45)
                    lines2.append(line)

            labels2.extend(labels)
            ax1.set_ylabel('Negative Log Likelihood', fontsize=plot_params['bfont'])
            ax1.set_ylim(plot_params['y2_range'])

    # Plot True IS MSE
    def plot_flat_line(ax, y, std, n, color='g', label='OIS', linestyle='--', linewidth=1):
        mean_mse = y * np.ones(length)
        std_mse = std * np.ones(length)
        yerr = Z_SCORE * std_mse / np.sqrt(n)
        x = np.arange(length) * plot_params['x_mult']
        line, = ax.plot(x, mean_mse, c=color, linestyle=linestyle, linewidth=linewidth)
        line.set_linewidth(3)
        ax.fill_between(x, mean_mse - yerr, mean_mse + yerr, facecolor=color, alpha=0.5)
        return line, label

    if plot_ois:
        true_IS_mse = np.array(true_IS_mse)
        line, label = plot_flat_line(ax2, np.mean(true_IS_mse), np.std(true_IS_mse),
                                     np.size(true_IS_mse))
        lines1.append(line)
        labels1.append(label)
    if plot_flat:
        line, label = plot_flat_line(ax2, flat_y, flat_std, flat_n, color='darkturquoise',
                                     label='Best Fit RIS')
        lines1.append(line)
        labels1.append(label)

    if plot_ois:
        true_IS_estimate = np.array(true_IS_estimate)
        line, label = plot_flat_line(ax3, np.mean(true_IS_estimate), np.std(true_IS_estimate),
                                     np.size(true_IS_estimate))
        lines3.append(line)
        labels3.append(label)
        line, label = plot_flat_line(ax3, [results.true_value], std=0, n=1, label='True Value',
                                     linestyle='-', linewidth=5)
        lines3.append(line)
        labels3.append(label)
    if plot_flat:
        line, label = plot_flat_line(ax3, flat_y, flat_std, flat_n, color='darkturquoise',
                                     label='Best Fit RIS')
        lines3.append(line)
        labels3.append(label)

    # Add arrow and vertical line showing where validation loss is minimized
    if plot_params['plot_min']:
        print(min_v_loss_point)
        ax1.annotate('Min validation loss', xy=(min_v_loss_point, 2.3),
                     xytext=(min_v_loss_point + 1000, 2.285), arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=25)
        params = {'c': 'k', 'linewidth': 4, 'linestyle': '--'}
        ax3.plot([min_v_loss_point, min_v_loss_point], [-10000, 10000], **params)
        ax2.plot([min_v_loss_point, min_v_loss_point], [-1000, 10000], **params)
        ax1.plot([min_v_loss_point, min_v_loss_point], [-1000, 10000], **params)

    # Set axis parameters
    if plot_params['x_range'] is not None:
        ax1.set_xlim(plot_params['x_range'])
    if plot_params['y1_range'] is not None:
        ax2.set_ylim(plot_params['y1_range'])
    if plot_params['y2_range'] is not None:
        ax1.set_ylim(plot_params['y2_range'])
    if plot_params['y3_range'] is not None:
        ax3.set_ylim(plot_params['y3_range'])

    ax3.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax1.yaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax2.yaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax3.yaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax2.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'])
    ax3.set_ylabel(plot_params['y3_label'], fontsize=plot_params['bfont'])
    ax2.set_ylabel(plot_params['y1_label'], fontsize=plot_params['bfont'])
    ax1.set_ylabel(plot_params['y2_label'], fontsize=plot_params['bfont'])

    # Add legend to each plot
    if plot_params['legend']:
        ax1.legend(lines2, labels2, fontsize=plot_params['lfont'],
                   loc=plot_params['legend_loc1'],
                   ncol=plot_params['legend_cols'])
        ax2.legend(lines1, labels1, fontsize=plot_params['lfont'],
                   loc=plot_params['legend_loc2'],
                   ncol=plot_params['legend_cols'])
        ax3.legend(lines3, labels3, fontsize=plot_params['lfont'],
                   loc=plot_params['legend_loc3'],
                   ncol=plot_params['legend_cols'])
    plt.show()


if __name__ == '__main__':
    main()
