"""Submit parallel trials to condor queueing system."""
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_directory', default='', help='Directory to write results to.')
parser.add_argument('--num_trials', 0, help='The number of trials to launch')
parser.add_argument('--trial_start', 0, help='Start seed')
parser.add_argument('--env', 'cartpole', help='Environment to run')
FLAGS = parser.parse_args()

TEST = False
# TEST = True
EXECUTABLE = '/u/jphanna/blackbox_importance_sampling/fit_dist_mse.py'
root = '/u/jphanna/blackbox_importance_sampling/'

args = {'--num_iters': 10000,
        '--batch_size': 400,
        '--eval_freq': 25,
        }

path = '%s/policies' % root
if FLAGS.env == 'hopper':
    env, env_path, pe, pb, pl = 'RoboschoolHopper-v1', 'hopper', 30, 20, 1000
elif FLAGS.env == 'cheetah':
    env, env_path, pe, pb, pl = 'RoboschoolHalfCheetah-v1', 'cheetah', 30, 20, 1000

args['--restore_path'] = '%s/%s/%s_%d' % (path, env_path, env_path, pe)
args['--behavior_path'] = '%s/%s/%s_%d' % (path, env_path, env_path, pb)
args['--env'] = env
args['--max_path_length'] = pl


def run_trial(seed, out_file, hidden_layers, hidden_units):

    arguments = '--result_file=%s --seed=%d' % (out_file, seed)

    if hidden_layers is not None:
        arguments += ' --hidden_layers=%d' % hidden_layers
    if hidden_units is not None:
        arguments += ' --hidden_units=%d' % hidden_units

    for arg in args:
        if args[arg] == '':
            arguments += ' %s' % arg
        else:
            arguments += ' %s=%s' % (arg, args[arg])

    cmd = '%s %s' % (EXECUTABLE, arguments)
    if TEST:
        print(cmd)
    else:
        subprocess.Popen(cmd.split())


def main():

    ct = 0
    directory = FLAGS.result_directory

    network_types = [(0, 0), (1, 64), (2, 64), (3, 64)]

    for network in network_types:

        for index in range(FLAGS.trial_start, FLAGS.num_trials):
            base = 'test'
            hidden_layers, hidden_units = network

            base += '_1'
            base += '_%d_%d' % network

            filename = os.path.join(directory, '%s_%d' % (base, index))
            if os.path.exists(filename):
                continue
            run_trial(index, filename, hidden_layers, hidden_units)
            ct += 1

    print('%d jobs submitted to cluster' % ct)


if __name__ == "__main__":
    main()
