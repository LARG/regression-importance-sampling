"""Launch gridworld experiments."""

from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random

parser = argparse.ArgumentParser()
parser.add_argument('result_directory', default=None, help='Directory to write results to.')
parser.add_argument('--num_trials', default=1, type=int, help='The number of trials to launch.')
parser.add_argument('--on-policy', default=False, action='store_true', help='Run on-policy experiment.')
FLAGS = parser.parse_args()

TEST = False
# TEST = True

EXECUTABLE = "./main"

exp_args = {
    '--domain': 0,
    '--iter': 1000,
    '--print_freq': 1000,
    '--iter-trajs': 10,
    # '--policy-number': 1,  # For on-policy experiments in paper
    # '--policy-number': 2,  # For off-policy experiments in paper
    '--behavior-number': 1
}


def run_trial(seed, outfile, per_decision=False, weighted=False, estimate=False,
              use_hold_out=False, estimate_with_all=False, use_control_variate=False,
              on_policy=False):
    """Run a single trial of a set method on Gridworld."""
    arguments = '--outfile %s --seed %d' % (outfile, seed)
    if per_decision:
        arguments += ' --per-decision'
    if weighted:
        arguments += ' --weighted'
    if estimate:
        arguments += ' --estimate'
    if use_control_variate:
        arguments += ' --use-cv'
    if use_hold_out:
        arguments += ' --hold-out'
    if estimate_with_all:
        arguments += ' --all-data'

    if on_policy:
        arguments += ' --policy-number 1'
    else:
        arguments += ' --policy-number 2'

    for arg in exp_args:
        arguments += ' %s %s' % (arg, exp_args[arg])

    cmd = '%s %s' % (EXECUTABLE, arguments)
    if TEST:
        print(cmd)
    else:
        subprocess.Popen(cmd.split())


class Method(object):
    """Object for holding method params."""

    def __init__(self, name, estimate=False, weighted=False, pd=False, cv=False, hold_out=False, alldata=False):  # noqa
        self.name = name
        self.estimate = estimate
        self.weighted = weighted
        self.per_decision = pd
        self.cv = cv
        self.hold_out = hold_out
        self.alldata = alldata


def main():  # noqa

    ct = 0
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    methods = []
    methods.append(Method('is', estimate=False, weighted=False, pd=False, cv=False, hold_out=False, alldata=False))  # IS
    methods.append(Method('wis', estimate=False, weighted=True, pd=False, cv=False, hold_out=False, alldata=False))  # WIS
    methods.append(Method('pdis', estimate=False, weighted=False, pd=True, cv=False, hold_out=False, alldata=False))  # PDIS
    methods.append(Method('is_dr', estimate=False, weighted=False, pd=False, cv=True, hold_out=False, alldata=False))  # DR
    methods.append(Method('is_wdr', estimate=False, weighted=True, pd=False, cv=True, hold_out=False, alldata=False))  # WDR
    methods.append(Method('ris', estimate=True, weighted=False, pd=False, cv=False, hold_out=False, alldata=False))  # RIS
    methods.append(Method('wris', estimate=True, weighted=True, pd=False, cv=False, hold_out=False, alldata=False))  # WRIS
    methods.append(Method('pdris', estimate=True, weighted=False, pd=True, cv=False, hold_out=False, alldata=False))  # PD-RIS
    methods.append(Method('ris_dr', estimate=True, weighted=False, pd=False, cv=True, hold_out=False, alldata=False))  # RIS DR
    methods.append(Method('ris_wdr', estimate=True, weighted=True, pd=False, cv=True, hold_out=False, alldata=False))  # RIS WDR
    methods.append(Method('ris_ho', estimate=True, weighted=False, pd=False, cv=False, hold_out=True, alldata=False))  # RIS
    methods.append(Method('ris_alldata', estimate=True, weighted=False, pd=False, cv=False, hold_out=False, alldata=True))  # RIS

    for seed in seeds:

        for method in methods:

            filename = os.path.join(directory, '%s_%d' % (method.name, seed))
            if os.path.exists(filename):
                continue
            run_trial(seed, filename, per_decision=method.per_decision,
                      weighted=method.weighted, estimate=method.estimate,
                      use_hold_out=method.hold_out, estimate_with_all=method.alldata,
                      use_control_variate=method.cv, on_policy=FLAGS.on_policy)
            ct += 1

    print('%d experiments ran.' % ct)


if __name__ == "__main__":
    main()
