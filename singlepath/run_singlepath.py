"""Run singlepath experiments."""

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
FLAGS = parser.parse_args()

TEST = False
# TEST = True

EXECUTABLE = 'singlepath.py'

args = {'--num_iters': 50000,
        '--eval_freq': 100}


def run_trial(seed, outfile):
    """Run single trial of singlepath experiment."""
    arguments = '%s --result_file=%s --seed=%d' % (EXECUTABLE, outfile, seed)

    for key in args:
        arguments += ' %s=%s' % (key, args[key])

    cmd = 'python %s' % arguments
    if TEST:
        print(cmd)
    else:
        subprocess.Popen(cmd.split())

def main():  # noqa

    ct = 0
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()
    directory = FLAGS.result_directory
    seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]

    for seed in seeds:
        filename = os.path.join(directory, 'trial_%d' % seed)
        if os.path.exists(filename):
            continue
        run_trial(seed, filename)
        ct += 1

    print('%d experiments ran.' % ct)


if __name__ == "__main__":
    main()
