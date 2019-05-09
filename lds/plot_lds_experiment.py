"""Plot policy evaluation experiment results."""

import argparse
import results_pb2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('result_directory', default='', help='Directory to write results to.', type=str)
parser.add_argument('--plot_file', default=None, help='File to save figure to.', type=str)
FLAGS = parser.parse_args()

rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

plot_params = {'bfont': 30,
               'lfont': 28,
               'tfont': 25,
               'legend': True,
               'legend_loc': 0,
               'legend_cols': 2,
               'y_range': [1e-0, 5e4],
               'x_range': [1e2, 1e4],
               'x_log_scale': True,
               'y_log_scale': True,
               'x_label': 'Number of Trajectories',
               'y_label': 'Mean Squared Error',
               'x_mult': 100}

methods_to_plot = ['ris_wdr_0.00', 'ris_wis_0.00', 'ris_is_0.00', 'pdis_0.00',
                   'wis_0.00', 'wdr_0.00', 'ris_pdis_0.00', 'ris_dr_0.00',
                   'dr_0.00', 'is_0.00']
methods_to_plot = None

plottype = 'gw-all-baselines'
plottype = 'gw-baselines'
# plottype = 'gw-alternatives'

if plottype == 'gw-alternatives':
    methods_to_plot = ['ris', 'is', 'ris_ho', 'ris_alldata']
elif plottype == 'gw-baselines':
    methods_to_plot = ['ris', 'wris', 'pdris', 'is', 'wis', 'pdis']
elif plottype == 'gw-all-baselines':
    methods_to_plot = ['ris', 'wris', 'pdris', 'is', 'wis', 'pdis']


class LineStyle(object):  # noqa

    def __init__(self, style, color, width, marker=None, markersize=None,  # noqa
                 dashes=None, alpha=0.5):
        self.color = color
        self.style = style
        self.width = width
        self.marker = marker
        self.markersize = markersize
        self.dashes = dashes
        self.alpha = alpha


def getLineStyle(label):  # noqa
    lower = label.lower()

    if plottype in ['gw-baselines', 'gw-all-baselines']:
        if 'dr' in lower and 'wdr' not in lower:
            style = LineStyle('-', 'c', 2, alpha=0.35)
            if 'ris' in lower:
                style.width = 5
                style.style = '--'
            return style
        if 'wdr' in lower:
            style = LineStyle(':', 'g', 3, alpha=0.35)
            if 'ris' in lower:
                style.width = 5
                style.style = '-.'
            return style
        if lower == 'ois' or lower == 'ris':
            style = LineStyle('--', 'b', 3, alpha=0.35)
            if 'ris' in lower:
                style.width = 5
                style.style = '-'
            return style
        if 'wis' in lower:
            style = LineStyle('-.', 'r', 3, alpha=0.35)
            if 'ris' in lower:
                style.width = 5
                style.style = '--'
            return style
        if 'pdis' in lower:
            style = LineStyle('--', 'k', 2, alpha=0.35)
            if 'ris' in lower:
                style.width = 5
                style.style = '-'
            return style
    else:
        if 'independent' in lower:
            return LineStyle('-.', 'c', 3, alpha=0.25)
        elif 'extra' in lower:
            return LineStyle(':', 'k', 4, alpha=0.25)
        elif lower.startswith('ris'):
            return LineStyle('-', 'b', 3, alpha=0.25)
        elif lower == 'ois':
            return LineStyle('--', 'r', 3, alpha=0.25)
        else:
            return LineStyle('--', 'k', 2, alpha=0.25)


def setLineStyle(label, line, itr=None):  # noqa
    style = getLineStyle(label)
    line.set_linestyle(style.style)
    if style.marker is not None:
        line.set_marker(style.marker)
    if style.markersize is not None:
        line.set_markersize(style.markersize)
    line.set_linewidth(style.width)
    if style.dashes is not None:
        line.set_dashes(style.dashes)
    if itr is not None and itr % 2 != 0:
        line.set_linestyle('--')


def read_proto(filename):  # noqa
    results = results_pb2.MethodResult()
    with open(filename, 'rb') as f:
        results.ParseFromString(f.read())
    return results


def get_label(name):  # noqa
    lower = name.lower()
    vals = lower.split('_')
    if 'ho' in vals:
        name = 'Independent Estimate'
    elif 'alldata' in vals:
        name = 'Extra-Data Estimate'
    elif 'ris' in vals[0]:
        name = 'RIS'
        if 'w' in vals[0]:
            name += ' WIS'
        if 'pd' in vals[0]:
            name += ' PDIS'
    elif 'is' in vals[0]:
        name = 'OIS'
        if 'w' in vals[0]:
            name += ' WIS'
        if 'pd' in vals[0]:
            name += ' PDIS'
    if 'dr' in vals:
        name += ' DR'
    if 'wdr' in vals:
        name += ' WDR'
    return name


def main():  # noqa

    if not FLAGS.result_directory:
        print ('No result directory given. Exiting.')
        return

    data = {}
    max_ind = 0
    best = 0.0

    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        if basename.endswith('.err'): continue
        try:
            results = read_proto(filename)
        except Exception as e:
            if 'Tag' not in str(e):
                raise e

        method = '_'.join(basename.split('_')[:-1])
        if len(results.estimates) == 0:
            continue
        if method not in data:
            print('new method', method)
            data[method] = []

        data[method].append(np.array(results.mse))

    print (max_ind, best)
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 12.0, forward=True)
    print(data.keys())
    print(methods_to_plot)
    if methods_to_plot is not None:
        methods = (x for x in methods_to_plot if x in data)
    else:
        methods = data.keys()

    for method in methods:
        n = np.size(data[method], axis=0)
        print (method, n)
        label = get_label(method)

        mean = np.mean(np.array(data[method]), axis=0)
        std = np.std(np.array(data[method]), axis=0)

        x = np.arange(np.size(mean)) * plot_params['x_mult']
        n = np.size(data[method], axis=0)

        yerr = 1.96 * std / np.sqrt(float(n))

        style = getLineStyle(label)
        color = style.color
        alpha = style.alpha
        line, = plt.plot(x, mean, label=label, c=color)
        plt.fill_between(x, mean - yerr, mean + yerr, facecolor=color,
                         alpha=alpha)
        setLineStyle(label, line)

    if plot_params['x_log_scale']:
        ax.set_xscale('log')
    if plot_params['y_log_scale']:
        ax.set_yscale('log')
    if plot_params['x_range'] is not None:
        plt.xlim(plot_params['x_range'])
    if plot_params['y_range'] is not None:
        plt.ylim(plot_params['y_range'])

    ax.xaxis.set_tick_params(labelsize=plot_params['tfont'])
    ax.yaxis.set_tick_params(labelsize=plot_params['tfont'])

    ax.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'])
    ax.set_ylabel(plot_params['y_label'], fontsize=plot_params['bfont'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])

    if FLAGS.plot_file is None:
        plt.show()
    else:
        fig.savefig(FLAGS.plot_file)


if __name__ == '__main__':
    main()
