"""Run RIS(n) and REG on singlepath domain."""

from __future__ import print_function
from __future__ import division

import argparse

import numpy as np
from matplotlib import pyplot as plt

import estimators
import mdps
import results_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='RNG Seed')
parser.add_argument('--num_iters', type=int, default=10000, help='Iterations to sample.')
parser.add_argument('--eval_freq', type=int, default=1, help='Frequency to estimate v(pi_e)')
parser.add_argument('--result_file', default=None, help='File to write results to.')
parser.add_argument('--plot', action='store_true', default=False, help='Plot trial results or not.')
FLAGS = parser.parse_args()


def eval_estimators(paths, idx, true_value,
                    is_method, is_estimates, is_mses, is_variances,
                    wis_method, wis_estimates, wis_mses,
                    reg_method, reg_estimates, reg_mses,
                    methods, estimates, mses, variances, labels):
    """Evaluate different estimators on the trajectory set."""
    is_estimates[idx], is_variances[idx] = is_method.estimate(paths)
    is_mses[idx] = (is_estimates[idx] - true_value) ** 2
    wis_estimates[idx], _ = wis_method.estimate(paths)
    wis_mses[idx] = (wis_estimates[idx] - true_value) ** 2
    reg_estimates[idx], _ = reg_method.estimate(paths)
    reg_mses[idx] = (reg_estimates[idx] - true_value) ** 2
    for i in range(len(methods)):
        estimates[i, idx], variances[i, idx] = methods[i].estimate(paths)
        mses[i, idx] = (estimates[i, idx] - true_value) ** 2
    print('Eval Number %d' % idx)
    print('IS MSE: %f' % is_mses[idx])
    print('WIS MSE: %f' % wis_mses[idx])
    print('REG MSE: %f' % reg_mses[idx])
    for i in range(len(methods)):
        print('%s MSE: %f' % (labels[i], mses[i, idx]))
    print('#############')


def main():  # noqa

    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)

    n_actions = 2
    horizon = 5
    n_methods = 2 * (horizon + 1)
    mdp = mdps.SinglePathMDP(n_actions, horizon, stochastic=True)
    pie = mdp.get_policy(0)
    pib = mdp.get_policy(1)
    print(pie)
    print(pib)
    paths = []

    results = results_pb2.Results()
    results.experiment_name = 'RIS_experiment'

    t_val = mdps.evaluate(mdp, pie)
    print('True value: %f' % t_val)

    IS = estimators.ISEstimate(pie, pib)
    WIS = estimators.ISEstimate(pie, pib, weighted=True)
    REG = estimators.REGEstimate(pie, mdp)
    methods = []
    labels = []
    for i in range(horizon):
        n = i + 1
        methods.append(estimators.RISEstimate(pie, n_actions, horizon, n=n))
        methods.append(estimators.RISEstimate(pie, n_actions, horizon, n=n,
                                              weighted=True))

        labels.append('RIS(%d)' % n)
        labels.append('Weighted RIS(%d)' % n)

    n_evals = int(FLAGS.num_iters / FLAGS.eval_freq)

    is_mses = np.zeros(n_evals)
    is_estimates = np.zeros(n_evals)
    is_variances = np.zeros(n_evals)
    wis_mses = np.zeros(n_evals)
    wis_estimates = np.zeros(n_evals)
    reg_mses = np.zeros(n_evals)
    reg_estimates = np.zeros(n_evals)
    mses = np.zeros((n_methods, n_evals))
    variances = np.zeros((n_methods, n_evals))
    estimates = np.zeros((n_methods, n_evals))
    lens = []
    idx = 0

    for itr in range(FLAGS.num_iters):
        path, G = mdps.sample(mdp, pib)

        paths.append(path)
        if itr % FLAGS.eval_freq == 0 and itr > 0:
            # idx = int(itr / FLAGS.eval_freq)
            eval_estimators(paths, idx, t_val,
                            IS, is_estimates, is_mses, is_variances,
                            WIS, wis_estimates, wis_mses,
                            REG, reg_estimates, reg_mses,
                            methods, estimates, mses, variances, labels)
            idx += 1

    eval_estimators(paths, idx, t_val, IS, is_estimates, is_mses, is_variances,
                    WIS, wis_estimates, wis_mses,
                    REG, reg_estimates, reg_mses,
                    methods, estimates, mses, variances, labels)

    # Normal IS methods
    method = results.methods.add()
    method.method_name = 'IS'
    method.estimates.extend(is_estimates)
    method.mse.extend(is_mses)
    method.variances.extend(is_variances)
    method = results.methods.add()
    method.method_name = 'WIS'
    method.estimates.extend(wis_estimates)
    method.mse.extend(wis_mses)
    method = results.methods.add()
    method.method_name = 'REG'
    method.mse.extend(reg_mses)
    method.estimates.extend(reg_estimates)

    # Add RIS methods
    for i in range(2 * horizon):
        method = results.methods.add()
        method.method_name = labels[i]
        method.estimates.extend(estimates[i])
        method.mse.extend(mses[i])
        method.variances.extend(variances[i])

    if FLAGS.result_file:
        with open(FLAGS.result_file, 'wb') as w:
            w.write(results.SerializeToString())

    # Plotting code
    if FLAGS.plot:
        lens = np.arange(n_evals) * FLAGS.eval_freq
        plt.plot(lens, is_mses, label='IS')
        plt.plot(lens, wis_mses, label='WIS')
        plt.plot(lens, reg_mses, label='REG')
        for i in range(2 * horizon):
            plt.plot(lens, mses[i], label=labels[i])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
