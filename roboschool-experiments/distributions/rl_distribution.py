import numpy as np
import gym
try:
    import roboschool
except ImportError as e:
    print(e)
from distributions import policies
from distributions import distributions


class ReinforcementLearning(distributions.Distribution):

    def __init__(self, env_str, policy_str,
                 max_path_length=None,
                 restore_path=None, scope=None, policy_args=None):
        super(ReinforcementLearning, self).__init__()
        self.env_str = env_str
        self.policy_str = policy_str
        self.env = gym.make(env_str)
        policy_cls = policies.get_policy_class(policy_str)

        self.policy = policy_cls(self.env.observation_space,
                                 self.env.action_space,
                                 scope=scope, **policy_args)
        self.max_path_length = max_path_length
        if restore_path is not None:
            self.policy.load_policy(restore_path)
        seed = None
        if policy_args is not None:
            seed = policy_args.get('seed', None)
        if seed is not None:
            print('Seeding RL Env')
            np.random.seed(seed)
            self.env.seed(seed)

    def _collect_paths(self, n_paths=1, render=False):

        paths = []
        for _ in range(n_paths):

            obs = self.env.reset()
            done = False
            t = 0
            path = {'observations': [],
                    'actions': [],
                    'rewards': [],
                    'action_probs': []}
            while not done:
                action, prob = self.policy.get_action(obs)
                if type(action) in [np.ndarray, np.array]:
                    action = action.flatten()
                n_obs, reward, done, _ = self.env.step(action)
                if type(reward) in [np.ndarray, list]:
                    reward = reward[0]
                if len(n_obs.shape) > 1:
                    n_obs = n_obs.flatten()
                if render:
                    self.env.render(mode='human')
                t += 1
                done = done or t == self.max_path_length
                path['observations'].append(obs)
                path['actions'].append(action)
                path['rewards'].append(reward)
                path['action_probs'].append(prob)
                obs = n_obs

            paths.append(path)

        return paths

    def sample(self):
        path = self._collect_paths(1)[0]
        ret = sum(path['rewards'])
        return path, ret

    def pdf(self, path):
        density = 1.
        for obs, action in zip(path['observations'], path['actions']):
            density *= self.policy.pdf(obs, action)
        return density

    def expected_value(self):
        n_eval_paths = 1000
        paths = self._collect_paths(n_eval_paths)
        returns = [sum(path['rewards']) for path in paths]
        return sum(returns) / float(n_eval_paths)

    def grad_log_dist(self, path, *args, **kwargs):
        grad_log_p = 0.0
        for obs, action in zip(path['observations'], path['actions']):
            grad_log_p += self.policy.grad_log_policy(obs, action)
        return grad_log_p

    def fit(self, samples, **kwargs):

        super(ReinforcementLearning, self).fit(samples, **kwargs)
        if len(samples[0]) == 2:
            paths = [sample[0] for sample in samples]
        else:
            paths = samples

        # 1. Get training data: obs -> action
        obs_list = []
        acts = []
        for path in paths:
            for obs, act in zip(path['observations'], path['actions']):
                obs_list.append(np.array(obs))
                acts.append(np.array(act))
        obs = np.array(obs_list)
        acts = np.array(acts)

        # 2. Initialize parameters of policy
        self.policy.initialize()
        # print(self.policy.session.run(self.policy.params))
        # self.policy.initialize()
        # print(self.policy.session.run(self.policy.params))
        n_steps = kwargs.get('n_itrs', 10000)
        batch_size = kwargs.get('batch_size', 200)
        inds = np.random.randint(np.size(obs, axis=0), size=200)
        test_obs, test_acts = obs[inds], acts[inds]

        # 3. fit policy
        for itr in range(n_steps):
            # inds = list(range(len(obs)))
            inds = np.random.randint(np.size(obs, axis=0), size=batch_size)
            obs_batch, acts_batch = obs[inds], acts[inds]
            self.policy.supervised_update(obs_batch, acts_batch)
            # if itr % 50 == 0:
            #     loss = self.policy.eval_loss(test_obs, test_acts)
            #     print (loss / 200.)

        return self.policy.eval_loss(test_obs, test_acts)
