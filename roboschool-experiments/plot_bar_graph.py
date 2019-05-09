"""Bar plots comparing MSE after training for different neural network architectures."""
import argparse
import numpy as np
from matplotlib import pyplot as plt

from matplotlib import rcParams
import os

from protos import blackbox_results_pb2


parser = argparse.ArgumentParser()
parser.add_argument('result_directory', default=None, help='Directory with results files.')
parser.add_argument('--plot_file', default=None, help='File to write plot to.')
FLAGS = parser.parse_args()

rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

Z_SCORE = 1.96

plot_params = {'bfont': 32,
               'lfont': 28,
               'xfont': 55,
               'yfont': 30,
               'legend': False,
               'legend_loc': 0,
               'legend_cols': 2,
               'y1_range': None,
               'y2_range': None,
               'bar_width': 0.35,
               # 'y1_range': [200, 700],  # Cheetah
               # 'y2_range': [5.8, 6],  # Cheetah
               'y1_range': [0, 700],  # Hopper
               'y2_range': [2.27, 2.28],  # Hopper
               'x_label': '',
               'y1_label': 'Mean Squared Error',
               'y2_label': 'Negative Log Likelihood'}

methods_to_plot = ['1-0-0', '1-1-64', '1-2-64', '1-3-64']


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

    mse_data = {}
    loss_data = {}
    true_IS_mse = []
    for basename in os.listdir(FLAGS.result_directory):
        filename = os.path.join(FLAGS.result_directory, basename)
        results = read_proto(filename)
        if results is None:
            continue
        method = '-'.join(basename.split('_')[1:-1])
        if method == '':
            continue

        if len(results.mse) == 0:
            continue
        if method not in mse_data:
            mse_data[method] = []
            loss_data[method] = []

        other_metric = np.array(results.validation_loss)
        best_ind = int(np.argmin(other_metric))
        mse_data[method].append(results.mse[best_ind])
        loss_data[method].append(other_metric[best_ind])

        if methods_to_plot is not None and method == methods_to_plot[0]:
            true_IS_mse.append(results.density_mse)
    print(list(mse_data.keys()))

    if methods_to_plot is not None:
        print(methods_to_plot)
        objects = ['-'.join(name.split('-')[1:])
                   for name in methods_to_plot]
        print(objects)
        objects = [name for name in methods_to_plot]
        objects += ['OIS']
        print(objects)
    else:
        objects = [key for key in mse_data] + ['OIS']
        print(objects)
    ris_methods = objects[:-1]  # everything but OIS
    mse = [np.mean(mse_data[key]) for key in ris_methods]
    mse_err = [np.std(mse_data[key]) for key in ris_methods]
    mse += [np.mean(true_IS_mse)]
    mse_err += [np.std(true_IS_mse)]
    loss = [np.mean(loss_data[key]) for key in ris_methods]
    best_model = np.argmin(loss)
    loss += [0.0]
    loss_err = [np.std(loss_data[key]) for key in ris_methods] + [0.0]

    # We divide by len(objects) - 1 for OIS because it has been computed
    # multiple times with same random seed.
    IS_count = int(np.size(true_IS_mse))
    if methods_to_plot is None:
        IS_count = int(np.size(true_IS_mse) / (len(objects) - 1))
    ns = ([np.size(mse_data[key]) for key in ris_methods] +
          [IS_count])
    objects = ['-'.join(obj.split('-')[1:]) for obj in objects[:-1]] +\
        ['OIS']
    print(objects)
    methods = {}
    methods['mse'] = blackbox_results_pb2.MethodCache()
    methods['loss'] = blackbox_results_pb2.MethodCache()
    names = ['mse', 'loss']
    means = [mse, loss]
    errs = [mse_err, loss_err]
    for name, mean, err in zip(names, means, errs):
        methods[name].mean.extend(mean)
        methods[name].err.extend(err)
        methods[name].n.extend(ns)
        methods[name].labels.extend(objects)

    # Print some of data statistics
    print('Statistics: n, mse, std, loss')
    for name, n, m, m_err, l in zip(objects, ns, mse, mse_err, loss):
        print ('\t', name, n, m, m_err, l)

    # Begin creating plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.set_size_inches(18, 16.0, forward=True)

    mse_err = [Z_SCORE / np.sqrt(n) * err for n, err in zip(ns, mse_err)]
    loss_err = [Z_SCORE / np.sqrt(n) * err for n, err in zip(ns, loss_err)]

    x_pos = np.arange(len(objects))

    if plot_params['y1_range'] is not None:
        ax1.set_ylim(plot_params['y1_range'])
    if plot_params['y2_range'] is not None:
        ax2.set_ylim(plot_params['y2_range'])

    # Add bars to plot
    bar_width = plot_params['bar_width']
    bars = ax1.bar(x_pos, mse, bar_width, yerr=mse_err, align='center', alpha=0.5,
                   label='MSE', color='b',
                   error_kw=dict(ecolor='b', lw=2, capsize=5, capthick=2))
    bars[best_model].set_hatch('//')
    bars[best_model].set_alpha(1.0)

    bars = ax2.bar(x_pos + bar_width, loss, bar_width, yerr=loss_err, align='center',
                   alpha=0.25, label='Loss', color='r',
                   error_kw=dict(ecolor='red', lw=2, capsize=5, capthick=2))
    bars[best_model].set_hatch('//')
    bars[best_model].set_alpha(1.0)

    # Label bars and set font size
    plt.xticks(x_pos, objects)
    ax1.xaxis.set_tick_params(labelsize=plot_params['xfont'])
    ax1.yaxis.set_tick_params(labelsize=plot_params['yfont'])
    ax2.yaxis.set_tick_params(labelsize=plot_params['yfont'])
    ax1.set_xlabel(plot_params['x_label'], fontsize=plot_params['bfont'])
    ax1.set_ylabel(plot_params['y1_label'], fontsize=plot_params['bfont'])
    ax2.set_ylabel(plot_params['y2_label'], fontsize=plot_params['bfont'])

    if plot_params['legend']:
        plt.legend(fontsize=plot_params['lfont'], loc=plot_params['legend_loc'],
                   ncol=plot_params['legend_cols'])
    plt.show()


if __name__ == '__main__':
    main()
