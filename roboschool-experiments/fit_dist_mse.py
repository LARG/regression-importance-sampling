import numpy as np
import tensorflow as tf
import argparse

from distributions import rl_distribution as rl
from distributions import policies
import replay_buffer as rb
from protos import blackbox_results_pb2
import common

parser = argparse.ArgumentParser()
parser.add_argument('--arg', help='Help!')

parser.add_argument('--print_freq', default=1, help='Frequency to print results.')
parser.add_argument('--eval_freq', default=1, help='Frequency to evaluate fit.')
parser.add_argument('--restore_path', default='', help='Path to restore model from.')
parser.add_argument('--behavior_path', default=None, help='Path to restore behavior policy from.')
parser.add_argument('--scope', default='', help='Scope for policy.')
parser.add_argument('--behavior_scope', default='', help='Scope for behavior policy.')
parser.add_argument('--num_iters', default=50, help='Number of policy training steps.')
parser.add_argument('--env', default='CartPole-v0', help='Environment to run')
parser.add_argument('--max_path_length', default=200, help='Maximum number of steps')
parser.add_argument('--seed', default=None, help='Seed for randomness')
parser.add_argument('--batch_size', default=400, help='Trajectories per iteration.')
parser.add_argument('--mini_batch_size', default=None, help='Training batch-size')
parser.add_argument('--result_file', default=None, help='File to write results to.')
parser.add_argument('--hidden_layers', default=0, help='Number of hidden layers for regression')
parser.add_argument('--hidden_units', default=None, help='Number of hidden units in each layer of regression net')
parser.add_argument('--target_layers', default=0, help='Number of hidden layers for target policy')
parser.add_argument('--target_units', default=None, help='Number of hidden units in each layer of target policy')
parser.add_argument('--behavior_layers', default=0, help='Number of hidden layers for behavior policy')
parser.add_argument('--behavior_units', default=None, help='Number of hidden units in each layer of behavior policy')
parser.add_argument('--entropy_coeff', default=None, help='Coefficient for entropy loss term')
parser.add_argument('--obs_len', default=1, help='Number of states and actions to concatenate.')
parser.add_argument('--weight_decay', default=0.02, help='L2 Regularization on MLE policy')
FLAGS = parser.parse_args()


def eval_target_policy(paths):
    est = 0.0
    var = 0.0
    ws = np.ones(len(paths))
    for ind, path in enumerate(paths):
        IS = sum(path['rewards']) * path['cumIWs'][-1]
        ws[ind] = path['cumIWs'][-1]
        est += IS
        var += IS ** 2

    return est / len(paths), var / len(paths)


def print_results(method, estimate, variance, mse):
    print('%s Estimate %f' % (method, estimate))
    print('%s Variance %f' % (method, variance))
    print('%s IS MSE %f' % (method, mse))


def add_results(results, is_estimate, is_variance, is_mse,
                entropy, loss, eval_loss, itr):
    results.estimates.append(is_estimate)
    results.variance.append(is_variance)
    results.mse.append(is_mse)
    results.entropy.append(float(entropy))
    results.losses.append(float(loss))
    results.validation_loss.append(float(eval_loss))
    results.iterations.append(itr)


def eval_pi_loss(policy, obs, acs):
    loss = policy.eval_loss(obs, acs)
    return loss / np.size(obs, axis=0)


def main():

    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)

    n_units = FLAGS.target_units if FLAGS.target_units is not None else 32
    hidden_sizes = [n_units for _ in range(FLAGS.target_layers)]
    policy_args = {'hidden_sizes': hidden_sizes}
    n_units = FLAGS.behavior_units if FLAGS.behavior_units is not None else 32
    hidden_sizes = [n_units for _ in range(FLAGS.behavior_layers)]
    behavior_args = {'hidden_sizes': hidden_sizes}

    true_value, _, avg_length, policy_args =\
        common.load_policy_args(FLAGS.restore_path, policy_args)

    if FLAGS.behavior_path is not None:
        _, _, _, behavior_args = common.load_policy_args(FLAGS.behavior_path,
                                                         behavior_args)
    else:
        _, _, _, behavior_args = common.load_policy_args(FLAGS.restore_path,
                                                         policy_args)

    policy_args['seed'] = FLAGS.seed
    behavior_args['seed'] = FLAGS.seed
    obs_len = FLAGS.obs_len
    print(policy_args, behavior_args)
    policy_str = 'Gaussian'
    if FLAGS.env == 'CartPole-v0':
        policy_str = 'boltzmann'
    distribution = rl.ReinforcementLearning(
        FLAGS.env, policy_str, max_path_length=FLAGS.max_path_length,
        scope=FLAGS.scope, policy_args=policy_args)
    behavior_dist = rl.ReinforcementLearning(
        FLAGS.env, policy_str, max_path_length=FLAGS.max_path_length,
        scope=FLAGS.behavior_scope, policy_args=behavior_args)

    if tf.gfile.Exists('%s.meta' % FLAGS.restore_path):
        distribution.policy.load_policy(FLAGS.restore_path)
    if FLAGS.behavior_path is not None:
        if tf.gfile.Exists('%s.meta' % FLAGS.behavior_path):
            behavior_dist.policy.load_policy(FLAGS.behavior_path)
    else:
        if tf.gfile.Exists('%s.meta' % FLAGS.restore_path):
            behavior_dist.policy.load_policy(FLAGS.restore_path)

    n_units = FLAGS.hidden_units if FLAGS.hidden_units is not None else 32
    hidden_sizes = [n_units for _ in range(FLAGS.hidden_layers)]
    mle_args = {'train_type': 'supervised',
                'seed': FLAGS.seed,
                'hidden_sizes': hidden_sizes,
                'entropy_coeff': FLAGS.entropy_coeff,
                'weight_decay': FLAGS.weight_decay,
                'act_fn': tf.nn.relu,
                'learning_rate': 1e-03}
    obs_space = distribution.env.observation_space
    act_space = distribution.env.action_space
    obs_space = common.elongate_space(obs_space, act_space, obs_len)
    if policy_str == 'Gaussian':
        policy_cls = policies.GaussianPolicy
        mle_args['learn_std'] = FLAGS.learn_std
    else:
        policy_cls = policies.ContinuousStateBoltzmannPolicy
    mle_policy = policy_cls(obs_space, act_space, scope='mle',
                            **mle_args)
    learning_iters = FLAGS.num_iters
    batch_size = FLAGS.batch_size
    ope_paths = []
    validation_paths = []
    replay_buffer = rb.ReplayBuffer()
    eval_buffer = rb.ReplayBuffer()

    results = blackbox_results_pb2.FitResults()
    results.method_name = 'nn_%d_%d' % (FLAGS.hidden_layers, n_units)

    replay_buffer.empty()
    eval_buffer.empty()

    # Get true value for eval policy. Either from file or with MC eval.
    if true_value is None:
        # if we couldn't load true value or we are using a mixture policy
        true_value = 0.0
        num_true_trajs = max(10000, 10 * batch_size)
        length = 0.0
        for _ in range(num_true_trajs):
            path, G = distribution.sample()
            true_value += G
            length += len(path['rewards'])
        true_value /= num_true_trajs
        avg_length = length / num_true_trajs
    results.true_value = true_value
    print('Avg path length %f' % avg_length)
    print('True value %f' % true_value)

    def mse(x):
        return (x - true_value) ** 2

    # Collect paths with behavior policy
    for _ in range(batch_size):
        path, G = behavior_dist.sample()
        replay_buffer.add_path(path, G)
        ope_paths.append(path)

    # Off-policy evaluation with true behavior policy
    common.load_importance_weights(distribution.policy, ope_paths, obs_len=1)
    is_estimate, is_variance = eval_target_policy(ope_paths, weighted=False)
    is_mse = mse(is_estimate)
    results.density_estimate = is_estimate
    results.density_variance = is_variance
    results.density_mse = is_mse
    print('###################')
    print('True Value %f' % true_value)
    print_results('True IS', is_estimate, is_variance, is_mse)

    pct_batch = 0.2
    for _ in range(int(pct_batch * batch_size)):
        path, G = behavior_dist.sample()
        validation_paths.append(path)

    # Get eval data
    _, _, eval_obs, eval_acts = common.get_train_test_data(validation_paths, split=0.0, obs_len=obs_len)
    train_obs, train_acts, _, _ = common.get_train_test_data(ope_paths, split=1.0, obs_len=obs_len)
    policy_eval_paths = ope_paths

    inds = np.arange(len(train_obs))

    common.load_importance_weights(distribution.policy, policy_eval_paths,
                                   mle_policy, obs_len=obs_len)
    entropy = mle_policy.entropy(train_obs)
    print('Entropy %f' % entropy)
    train_loss = eval_pi_loss(mle_policy, train_obs, train_acts)
    eval_loss = eval_pi_loss(mle_policy, eval_obs, eval_acts)

    is_estimate, is_variance = eval_target_policy(policy_eval_paths,
                                                  weighted=False)
    is_mse = mse(is_estimate)

    print_results('RIS', is_estimate, is_variance, is_mse)
    print('Training Loss %f' % train_loss)
    print('Validation Loss %f' % eval_loss)
    print('Entropy %f' % entropy)
    print('###################')
    add_results(results, is_estimate, is_variance, is_mse, entropy, train_loss, eval_loss, 0)

    for itr in range(learning_iters):

        if FLAGS.mini_batch_size is not None:
            m = len(train_obs)
            inds = np.random.randint(m, size=FLAGS.mini_batch_size)

        obs_batch, acts_batch = train_obs[inds], train_acts[inds]

        # loss is training error, v_loss is validation error computed on
        # samples that we will also use in policy evaluation, eval_loss is
        # validation loss on samples that will not be used in
        # policy evaluation.
        mle_policy.supervised_update(obs_batch, acts_batch)
        train_loss = eval_pi_loss(mle_policy, train_obs, train_acts)
        eval_loss = eval_pi_loss(mle_policy, eval_obs, eval_acts)
        entropy = mle_policy.entropy(train_obs)

        if itr > 0 and itr % FLAGS.eval_freq == 0:
            common.load_importance_weights(distribution.policy, policy_eval_paths,
                                           mle_policy, obs_len=obs_len)
            is_estimate, is_variance = eval_target_policy(policy_eval_paths, weighted=False)

            is_mse = mse(is_estimate)

            if itr % FLAGS.print_freq == 0:
                print('###################')
                print('Iteration %d' % itr)
                print_results('RIS', is_estimate, is_variance, is_mse)
                print('Training Loss %f' % train_loss)
                print('Validation Loss %f' % eval_loss)
                print('Entropy %f' % entropy)
                print('###################')
            add_results(results, is_estimate, is_variance, is_mse,
                        entropy, train_loss, eval_loss, itr)

    if FLAGS.result_file is not None:
        with open(FLAGS.result_file, 'wb') as w:
            w.write(results.SerializeToString())


if __name__ == '__main__':
    main()
