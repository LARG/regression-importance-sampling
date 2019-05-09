import sys
import numpy as np
import scipy.signal
import tensorflow as tf
from gym import spaces

from protos import blackbox_results_pb2


def discount_cumsum(rewards, gamma):
    return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1],
                                axis=0)[::-1]


def elongate_space(obs_space, act_space, n):
    """
    Gets the space of type: [obs, act, obs, act, ..., obs]

    Args:
    obs_space: gym.spaces.Box
               Box space object for observations
    act_space: gym.spaces.Box or gym.spaces.Discrete
               Space object for actions
    n: int, the number of times to repeat obs and act + 1.

    Returns:
        spaces.Box, new spaces.Box object
    """
    if n == 1:
        return obs_space
    a_space = act_space
    if isinstance(act_space, spaces.Discrete):
        a_space = spaces.Box(0, 1, shape=(1))
    new_low = np.concatenate([obs_space.low, a_space.low], axis=0)
    new_high = np.concatenate([obs_space.high, a_space.high], axis=0)
    new_low = np.tile(new_low, n - 1)
    new_high = np.tile(new_high, n - 1)
    new_low = np.concatenate([new_low, obs_space.low], axis=0)
    new_high = np.concatenate([new_high, obs_space.high], axis=0)

    return spaces.Box(np.array(new_low), np.array(new_high))


def load_importance_weights(policy, paths, mle_policy=None, obs_len=1,
                            load_eval_probs=True):
    """
    Loads importance weights for target policy into each path in paths.

    Args:
    policy: policies.Policy
            The target policy to load into eval_probs key of each path
    paths: List of dicts
            The paths to load importance weights for.
    mle_policy: policies.Policy
                Optional policy to load behavior weights for.
                WARNING: this overwrites the behavior probabilities of the true
                behavior policy
    obs_len: The number of observations and actions to concatenate for input
             to the mle_policy.

    Returns:
        None
    """
    obs_dim = np.size(paths[0]['observations'][0])
    act_dim = np.size(paths[0]['actions'][0])
    for path in paths:
        if load_eval_probs:
            path['eval_probs'] = []
        path['IWs'] = []
        path['cumIWs'] = []
        cumObs = np.concatenate([np.zeros(obs_dim), np.zeros(act_dim)])
        cumObs = np.tile(cumObs, obs_len - 1)
        for t, obs in enumerate(path['observations']):
            action = path['actions'][t]
            action_prob = path['action_probs'][t]
            cumObs = np.concatenate([cumObs.flatten(), obs], axis=0)
            if mle_policy is not None:
                action_prob = mle_policy.pdf(cumObs, action)
            if isinstance(action, int):
                cumObs = np.concatenate([cumObs, np.array([action])], axis=0)
            else:
                cumObs = np.concatenate([cumObs, action], axis=0)
            cumObs = cumObs[obs_dim + act_dim:]
            if load_eval_probs:
                path['eval_probs'].append(policy.pdf(obs, action))
            path['IWs'].append(path['eval_probs'][t] / action_prob)
            if t == 0:
                path['cumIWs'].append(path['IWs'][t])
            else:
                path['cumIWs'].append(path['IWs'][t] * path['IWs'][t - 1])


def fit_policy(policy, paths, n_steps=10000, batch_size=200,
               eval_freq=100, stop_criterion='holdout',
               run_twice=False, obs_len=1, cv_folds=None):

    if cv_folds is not None:
        all_obs, all_acts, _, _ = get_train_test_data(
            paths, split=1.0, obs_len=obs_len)
        fold_size = int(np.size(all_obs, axis=0) / cv_folds)
        losses = []
        for i in range(cv_folds):
            inds = np.arange(0, i * fold_size).tolist()
            inds += np.arange((i + 1) * fold_size,
                              np.size(all_obs, axis=0)).tolist()
            obs, acts = all_obs[inds], all_acts[inds]
            cutoff = int(np.size(obs, axis=0) * 0.8)
            train_obs, train_acts, eval_obs, eval_acts =\
                obs[:cutoff], acts[:cutoff], obs[cutoff:], acts[cutoff:]
            fit(policy, train_obs, train_acts, eval_obs, eval_acts, n_steps,
                batch_size, eval_freq, stop_criterion, run_twice, obs_len)
            inds = np.arange(i * fold_size, (i + 1) * fold_size).tolist()
            obs, acts = all_obs[inds], all_acts[inds]
            loss = policy.eval_loss(obs, acts) / np.size(obs, axis=0)
            losses.append(loss)
            print('Fold %d: %f' % (i, loss))
        train_obs, train_acts, eval_obs, eval_acts = get_train_test_data(
            paths, obs_len=obs_len)
        fit(policy, train_obs, train_acts, eval_obs, eval_acts, n_steps,
            batch_size, eval_freq, stop_criterion, run_twice, obs_len)
        return np.mean(losses)
    else:
        train_obs, train_acts, eval_obs, eval_acts = get_train_test_data(
            paths, obs_len=obs_len)
        fit(policy, train_obs, train_acts, eval_obs, eval_acts, n_steps,
            batch_size, eval_freq, stop_criterion, run_twice, obs_len)
        return policy.eval_loss(eval_obs, eval_acts)


def fit(policy, train_obs, train_acts, eval_obs, eval_acts, n_steps=1e4,
        batch_size=200, eval_freq=100, stop_criterion='holdout',
        run_twice=False, obs_len=1):

    inds = np.arange(len(train_obs))
    train_steps = n_steps
    iters = 2 if run_twice else 1
    for i in range(iters):
        train_steps = int(train_steps / 2 ** i)
        policy.initialize()
        prev_loss = None
        meta = {'tloss': [], 'vloss': [], 'mse': []}

        for itr in range(train_steps):

            if batch_size is not None:
                inds = np.random.randint(len(train_obs), size=batch_size)
            obs_batch, acts_batch = train_obs[inds], train_acts[inds]
            loss = policy.supervised_update(obs_batch, acts_batch)
            meta['tloss'].append(loss)

            if itr % eval_freq == 0 and stop_criterion == 'holdout':
                loss = policy.eval_loss(eval_obs, eval_acts)
                # print (itr, loss)
                meta['vloss'].append(loss)
                if prev_loss is not None and loss > prev_loss:
                    train_steps = itr
                    break
                prev_loss = loss
    return meta


def get_train_test_data(paths, split=0.8, obs_len=1):
    """
    Gets train and test data of (obs -> acs) from paths.

    Args:
        paths: List of dicts
                Paths to divide into train / test data
        split: float
                Percentage of data to be used for training vs. testing
        obs_len:
                Length of obs, action history to keep around for observations.

    Returns:
        train_obs: np.array of observations for training
        train_acts: np.array of actions as training targets
        test_obs: np.array of observations for testing
        test_acts: np.array of actions as testing targets
    """

    obs_dim = np.size(paths[0]['observations'][0])
    act_dim = np.size(paths[0]['actions'][0])
    # 1. Get training data: obs -> action
    obs_list = []
    acts = []
    for path in paths:
        cumObs = np.concatenate([np.zeros(obs_dim), np.zeros(act_dim)])
        cumObs = np.tile(cumObs, obs_len - 1)
        for obs, act in zip(path['observations'], path['actions']):
            cumObs = np.concatenate([cumObs.flatten(), obs], axis=0)
            obs_list.append(np.array(cumObs))
            acts.append(np.array(act))
            if isinstance(act, int):
                cumObs = np.concatenate([cumObs, np.array([act])], axis=0)
            else:
                cumObs = np.concatenate([cumObs, act], axis=0)
            cumObs = cumObs[obs_dim + act_dim:]
    obs = np.array(obs_list)
    acts = np.array(acts)

    # 2. Get validation data: 80-20 train-validation split
    n = np.size(obs, axis=0)
    split = min(max(0.0, split), 1.0)
    train_end = int(split * n)
    train_obs = obs[:train_end]
    train_acts = acts[:train_end]
    eval_obs = obs[train_end:]
    eval_acts = acts[train_end:]

    return train_obs, train_acts, eval_obs, eval_acts


def _load_from_args_file(filename):
    args = {}
    with open(filename, 'r') as o:
        line = o.readline()
        while line:
            key, arg = line.split(': ')
            l = len(arg)
            assert l < 12, 'Arg should not be too long: %s %d' % (arg, l)
            args[key] = eval(arg)
            line = o.readline()
    hidden_units = args.get('hidden_units', 0)
    hidden_layers = args.get('hidden_layers', 0)
    args.pop('hidden_units')
    args.pop('hidden_layers')
    args['hidden_sizes'] = [hidden_units for _ in range(hidden_layers)]
    return args


def load_policy_args(restore_path, default_args={}):

    if tf.gfile.Exists('%s.info' % restore_path):
        filename = '%s.info' % restore_path
        info = blackbox_results_pb2.PolicyData()
        args = {}
        try:
            with open(filename, 'rb') as f:
                info.ParseFromString(f.read())
        except Exception:
            pass
        else:
            h_units = info.hidden_units
            h_layers = info.hidden_layers
            args['hidden_sizes'] = [h_units for _ in range(h_layers)]
            args['filter_obs'] = info.filter_obs
            assert len(info.act_fn) < 12, 'Act fn too long'
            try:
                args['act_fn'] = eval(info.act_fn)
            except NameError:
                args['act_fn'] = tf.nn.relu
            ret = info.average_return
            var = info.average_variance
            p_len = info.average_path_length
            print('Loaded policy args from %s.info file' % restore_path)
            return ret, var, p_len, args
    if tf.gfile.Exists('%s.args' % restore_path):
        filename = '%s.args' % restore_path
        print('Loaded policy args from %s.args file' % restore_path)
        return None, None, None, _load_from_args_file(filename)
    else:
        print('Failed to load args, returning defaults')
        return None, None, None, default_args


def set_global_seeds(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def randint(n, size=1):
    return np.random.randint(n, size=size)


def randnorm(size=1):
    return np.random.normal(size=size)


def randchoice(n, p=None):
    return np.random.choice(n, p=p)


if __name__ == '__main__':
    args = load_policy_args(sys.argv[1])
    print(args)
