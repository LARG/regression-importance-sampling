"""Launch LDS experiments."""

from __future__ import print_function
from __future__ import division

import os
import subprocess

import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', default=None, help='Directory to write results to.')
parser.add_argument('--num_trials', type=int, default=0, help='The number of trials to launch')
parser.add_argument('--learn_std', action='store_true', default=True, help='Learn Standard deviation or not.')
parser.add_argument('--no-learn_std', action='store_false', dest='learn_std', default=True, help='Learn Standard deviation or not.')
parser.add_argument('--dist_type', default='lds', help='Distribution type')
parser.add_argument('--max_poly_deg', default=1, type=int, help='Degree polynomial to fit.')
parser.add_argument('--fit_type', default='mle', help='MLE or gradient ascent fit. Or None.')
parser.add_argument('--weighted', action='store_true', default=False, help='Weighted IS or not.')
parser.add_argument('--true_poly_deg', default=1, type=int, help='Degree for pi_b basis functions.')
parser.add_argument('--use_hold_out', action='store_true', default=False, help='Estimate pib with separate data.')
FLAGS = parser.parse_args()

TEST = False
# TEST = True

EXECUTABLE = 'policy_eval_experiment.py'

args = {'--num_samples': 10000,
        '--eval_freq': 100}


def run_trial(seed, outfile, dist_type='lds', per_decision=False, weighted=False,
              estimate=False, use_hold_out=False, estimate_with_all=False):
    """Launch single trial of specified method."""
    arguments = '--result_file=%s --seed=%d' % (outfile, seed)
    arguments += ' --dist_type=%s' % dist_type

    if per_decision:
        arguments += ' --per-decision'
    if weighted:
        arguments += ' --weighted'
    if estimate:
        arguments += ' --estimate'

    if use_hold_out:
        arguments += ' --hold-out'
    if estimate_with_all:
        arguments += ' --all-data'

    for arg in args:
        arguments += ' %s=%s' % (arg, args[arg])

    cmd = 'python %s %s' % (EXECUTABLE, arguments)
    if TEST:
        print(cmd)
    else:
        subprocess.Popen(cmd.split())


class Method(object):
    """Parameters of methods to run."""

    def __init__(self, name, estimate=False, weighted=False, pd=False, hold_out=False, alldata=False):  # noqa
        self.name = name
        self.estimate = estimate
        self.weighted = weighted
        self.per_decision = pd
        self.hold_out = hold_out
        self.alldata = alldata


def main():  # noqa

    ct = 0
    directory = FLAGS.result_directory
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    methods = []
    methods.append(Method('is', estimate=False, weighted=False, pd=False, hold_out=False, alldata=False))  # IS
    methods.append(Method('wis', estimate=False, weighted=True, pd=False, hold_out=False, alldata=False))  # WIS
    methods.append(Method('pdis', estimate=False, weighted=False, pd=True, hold_out=False, alldata=False))  # PDIS
    methods.append(Method('ris', estimate=True, weighted=False, pd=False, hold_out=False, alldata=False))  # RIS
    methods.append(Method('wris', estimate=True, weighted=True, pd=False, hold_out=False, alldata=False))  # WRIS
    methods.append(Method('pdris', estimate=True, weighted=False, pd=True, hold_out=False, alldata=False))  # PD-RIS
    methods.append(Method('ris_ho', estimate=True, weighted=False, pd=False, hold_out=True, alldata=False))  # Independent estimate
    methods.append(Method('ris_alldata', estimate=True, weighted=False, pd=False, hold_out=False, alldata=True))  # Extra-data estimate

    for seed in seeds:

        for method in methods:

            filename = os.path.join(directory, '%s_%d' % (method.name, seed))
            if os.path.exists(filename):
                continue
            run_trial(seed, filename, per_decision=method.per_decision,
                      weighted=method.weighted, estimate=method.estimate,
                      use_hold_out=method.hold_out, estimate_with_all=method.alldata)
            ct += 1

    print('%d experiments ran.' % ct)


if __name__ == "__main__":
    main()
