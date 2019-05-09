import numpy as np


class Normal(object):

    def __init__(self, mean, sigma):
        self._mean = mean
        self._sigma = sigma
        self._dim = 1

    def sample(self, n=1):
        xs = np.random.normal(self._mean, self._sigma, size=n)
        return xs, np.array(xs)

    @property
    def mean(self):
        return self._mean

    def mle_fit(self, xs):
        self._mean = np.mean(xs)
        self._sigma = np.std(xs)

    def _pdf(self, x):
        if self._sigma == 0:
            if x == self._mean:
                return 1.0
            else:
                return 0.0
        const = 1 / (2 * np.pi * self._sigma ** 2) ** (0.5)
        diff = x - self._mean
        return const * np.exp(-1 * diff ** 2 / (2 * self._sigma ** 2))

    def pdf(self, xs):
        return np.vectorize(self._pdf)(xs)

    def nlh(self, xs):
        if self._sigma == 0:
            if xs == self._mean:
                return 0.0
            else:
                return -1e6
        log_std = np.log(self._sigma)
        zs = (xs - self._mean) / self._sigma
        nlh = (log_std + 0.5 * np.square(zs) +
               0.5 * self._dim * np.log(2 * np.pi))
        return np.sum(nlh)

