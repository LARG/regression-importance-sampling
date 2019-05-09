import numpy as np
from scipy import special


def is_ratio(x, sampling_distribution, target_distribution):
    num = target_distribution.pdf(x)
    denom = sampling_distribution.pdf(x)
    return num / denom


def is_hashable(x):
    return not isinstance(x, list) and not isinstance(x, dict)


def to_hashable(x):
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, dict):
        hashable = tuple()
        for key in x:
            hashable += tuple(np.array(x[key]).flatten().tolist())
        return hashable
    return x


class Distribution(object):

    def __init__(self):
        self.fitted = False

    def sample(self):
        pass

    def pdf(self, x):
        pass

    def expected_value(self):
        pass

    def grad_log_dist(self, x, *args, **kwargs):
        pass

    def grad_log_dist_as_vec(self, x):
        pass

    def fit(self, samples, **kwargs):
        self.fitted = True


class Normal(Distribution):

    def __init__(self, mean=0, std_dev=1):
        super(Normal, self).__init__()
        self._mean = mean
        self._std_dev = std_dev

    def _f(self, x):
        return x

    def sample(self, n=1):
        x = np.random.normal(self._mean, self._std_dev)
        return x, self._f(x)

    def grad_log_dist(self, x, param='mean'):
        s = self._std_dev
        if param == 'mean':
            return -2 * (x - self._mean) * -1 / s ** 2
        elif param == 'std':
            return -1.0 / s + ((x - self._mean) ** 2) / float(s ** 3)
        else:
            raise NotImplementedError('Can only use mean')

    def grad_log_dist_as_vec(self, x):
        params = ['mean', 'std']
        return np.array([self.grad_log_dist(x, param=p) for p in params])

    def pdf(self, x):
        const = 1 / (2 * np.pi * self._std_dev ** 2) ** (0.5)
        diff = x - self._mean
        return const * np.exp(-1 * diff ** 2 / (2 * self._std_dev ** 2))

    def expected_value(self):
        return self._mean

    @property
    def support_type(self):
        return 'continuous'

    def fit(self, samples, **kwargs):
        xs = np.array([x[0] for x in samples])
        if 'weighted_fit' not in kwargs:
            self._mean = np.mean(xs)
            self._std_dev = np.std(xs) + 1e-8
        else:
            ws = np.array([x[1] for x in samples])
            self._mean = xs.dot(ws) / np.sum(ws)
            self._std_dev = np.std(xs) + 1e-8


class BanditArms(object):

    def __init__(self, arm_payoffs, arm_means=None):
        """
        arm_payoffs: array of scalars or callables.
        arm_means: the expected value of pulling an arm.
        """
        self.arm_payoffs = arm_payoffs
        if arm_means:
            self.arm_means = arm_means
        else:
            self.arm_means = arm_payoffs

    def pull(self, arm):
        if self.arm_payoffs[arm] is callable:
            return self.arm_payoffs[arm]()
        return self.arm_payoffs[arm]

    def expected_value(self, arm_probs):
        assert len(arm_probs) == len(self.arm_means)
        exp_val = sum([prob * value for prob, value in zip(arm_probs,
                                                           self.arm_means)])
        return exp_val


class BoltzmannBandit(Distribution):

    def __init__(self, bandit, dist_params):
        """
        bandit: BanditArms.
          The arms to pull and receive rewards from.
        dist_params: float array.
          Parameters of boltzmann distribution.
        """
        super(BoltzmannBandit, self).__init__()
        self._bandit = bandit
        self._arm_probs = np.exp(dist_params) / np.sum(np.exp(dist_params))

    @property
    def n_arms(self):
        return len(self._arm_probs)

    @property
    def bandit(self):
        return self._bandit

    def sample(self):
        arm = np.random.choice(self.n_arms, p=self._arm_probs)
        return arm, self._bandit.pull(arm)

    def pdf(self, x):
        return self._arm_probs[x]

    def grad_log_dist(self, x, param=0):
        if x == param:
            return 1 - self.pdf(x)
        return -1 * self.pdf(x)

    def grad_log_dist_as_vec(self, x):
        dist = -1 * self._arm_probs
        dist[x] += 1
        return dist

    def expected_value(self):
        return self._bandit.expected_value(self._arm_probs)

    @property
    def support_type(self):
        return 'discrete'

    def fit(self, samples, **kwargs):
        super(BoltzmannBandit, self).fit(samples, **kwargs)
        weights = kwargs.get('sample_weights', np.ones(len(samples)))
        sample_counts = np.zeros(self.n_arms)
        xs = [x[0] for x in samples]
        for i, x in enumerate(xs):
            if not isinstance(x, (int, np.int64)):
                print('[BoltzmannBandit.fit]: '
                      'samples contained non-integer sample {}'.format(x))
                continue
            sample_counts[x] += 1 * weights[i]
        n_samples = float(len(xs))
        self._arm_probs = np.array(sample_counts) * 1. / n_samples


class Beta(Distribution):

    def __init__(self, alpha=0, beta=1):
        self._alpha = alpha
        self._beta = beta
        super(Beta, self).__init__()

    def _f(self, x):
        return x

    def sample(self, n=1):
        x = np.random.beta(self._alpha, self._beta)
        return x, self._f(x)

    def grad_log_dist(self, x, param='alpha'):
        digamma_sum = special.digamma(self._alpha + self._beta)
        if param == 'alpha':
            return np.log(x) + digamma_sum - special.digamma(self._alpha)
        elif param == 'beta':
            return np.log(1 - x) + digamma_sum - special.digamma(self._beta)
        else:
            raise NotImplementedError

    def pdf(self, x):
        B = (special.gamma(self._alpha) * special.gamma(self._beta) /
             special.gamma(self._alpha + self._beta))
        return (x ** (self._alpha - 1) * (1 - x) ** (self._beta - 1)) / B

    def expected_value(self):
        return self._alpha / (self._alpha + self._beta)

    @property
    def support_type(self):
        return 'continuous'

    def fit(self, samples, **kwargs):
        raise NotImplementedError


class Exponential(Distribution):

    def __init__(self, scale=1):
        super(Exponential, self).__init__()
        self._lambda = scale ** (-1)

    def _f(self, x):
        return x

    def sample(self, n=1):
        x = np.random.exponential(self._lambda ** (-1))
        return x, self._f(x)

    def grad_log_dist(self, x, param=None):
        return self._lambda ** (-1) - x

    def pdf(self, x):
        return self._lambda * np.exp(-1 * self._lambda * x)

    def expected_value(self):
        return self._lambda ** (-1)

    @property
    def support_type(self):
        return 'continuous'

    def fit(self, samples, **kwargs):
        super(Exponential, self).fit(samples, **kwargs)
        xs = np.array([x[0] for x in samples])
        mean = np.mean(xs)
        self._lambda = mean ** (-1)
