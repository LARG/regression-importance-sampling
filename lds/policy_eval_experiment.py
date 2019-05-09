"""This script runs a policy evaluation metric with increasing data."""

import argparse

import numpy as np

from lds import LDS
from normal import Normal
import common
from policies import lds_policies
import results_pb2


def bool_argument(parser, name, help=''):
    """Helper function for adding boolean arguments."""
    parser.add_argument('--%s' % name, action='store_true', default=False, help=help)
    parser.add_argument('--no-%s' % name, action='store_false', dest=name, help=help)

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to draw.')
parser.add_argument('--seed', type=int, default=None, help='RNG Seed')
parser.add_argument('--eval_freq', type=int, default=1, help='How often to compute estimate.')
parser.add_argument('--print_freq', type=int, default=1, help='How often to print results.')
bool_argument(parser, 'estimate', help='Estimate behavior policy or not.')
bool_argument(parser, 'weighted', help='Use weighted IS.')
bool_argument(parser, 'per-decision', help='Use per-decision IS.')
bool_argument(parser, 'hold-out', help='Use separate data for pib estimate.')
bool_argument(parser, 'all-data', help='Use extra data for pib estimate.')
parser.add_argument('--result_file', default=None, help='File to write results to.')
parser.add_argument('--dist_type', default='lds', help='Distribution type for experiment.')
FLAGS = parser.parse_args()


def get_dists(dist_type, estimate):

    if dist_type == 'lds':
        p_std, q_std = 0.5, 0.6
        true_degree = list(range(3))
        poly_deg = list(range(3))
        in_dim = len(true_degree) * 2 * 2
        q_k = lds_policies['lds_2_2_near_final']
        p_k = lds_policies['lds_2_2_near_final']

        # target distribution (evaluation policy)
        p = LDS(p_k, p_std, polynomial_degrees=true_degree)
        # sampling distribution (behavior policy)
        q = LDS(q_k, q_std, polynomial_degrees=true_degree)

        in_dim = len(poly_deg) * 2 * 2
        init_k = np.ones((in_dim, 2))
        if not estimate:
            init_k = np.array(q_k, copy=True)
            poly_deg = true_degree

        fit_q = LDS(init_k, q_std, polynomial_degrees=poly_deg)
    else:
        p = Normal(1, 1)
        q = Normal(0, 1)
        fit_q = Normal(0, 1)
    return p, q, fit_q


def main():

    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)

    # Results proto
    results = results_pb2.MethodResult()

    # Create distributions and sample
    p, q, fit_q = get_dists(FLAGS.dist_type, FLAGS.estimate)
    true_value = p.expected_value()

    def mse(x):
        return (true_value - x) ** 2

    # IS estimators
    estimator = common.ISEstimator(p, fit_q, weighted=FLAGS.weighted, per_decision=FLAGS.per_decision)

    print('###################')
    print('True Value %f' % true_value)
    print('###################')

    results.true_value = true_value

    xs, fs = [], []
    ho_xs = []

    for itr in range(FLAGS.num_samples):

        # Sample data for policy evaluation
        x, f = q.sample()

        # Sample "unlabelled" data (no return info)
        ho_x, _ = q.sample()
        if type(x) is list:
            xs.extend(x)
            ho_xs.extend(ho_x)
        else:
            xs.append(x)
            ho_xs.append(ho_x)
        fs.append(f)

        # Choose fit_xs and eval_xs:
        # eval means we fit with policy evaluation data
        # hold-out means we fit with a separate data set
        # otherwise we aggregate data for fitting.
        if FLAGS.estimate:
            fit_xs = xs
            if FLAGS.hold_out:
                fit_xs = ho_xs
            elif FLAGS.all_data:
                fit_xs = xs + ho_xs

        if itr % FLAGS.eval_freq == 0 or itr == FLAGS.num_samples:

            # 1. Fit distribution
            if FLAGS.estimate:
                fit_q.mle_fit(fit_xs)

            # 2. Get Estimate and MSE
            estimate = estimator.estimate(xs, np.array(fs).flatten())
            est_mse = mse(estimate)

            # 4. Output diagnostics to terminal
            if itr % FLAGS.print_freq == 0 or itr == FLAGS.num_samples:
                print('###################')
                print('Iteration %d' % itr)
                print('Estimate %f' % estimate)
                print('MSE %f' % est_mse)
                print('###################')

            # 5. Record results
            results.estimates.append(estimate)
            results.mse.append(est_mse)
            results.num_samples.append(float(len(fs)))

    if FLAGS.result_file is not None:
        with open(FLAGS.result_file, 'wb') as w:
            w.write(results.SerializeToString())


if __name__ == '__main__':
    main()
