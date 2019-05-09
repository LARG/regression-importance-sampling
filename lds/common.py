"""Utility code for LDS experiments."""

import numpy as np


class ISEstimator(object):

    def __init__(self, target_dist, sample_dist, weighted=False,
                 per_decision=False):
        self._target_dist = target_dist
        self._sample_dist = sample_dist
        self._weighted = weighted
        self._per_decision = per_decision

    def estimate(self, xs, fs):
        if self._per_decision and type(xs[0]) is dict:
            ps = self._target_dist.pdf_per_decision(xs)
            qs = self._sample_dist.pdf_per_decision(xs)
            ratios = ps / qs
            Rs = np.array([path['r'] for path in xs])
            pdis = np.sum(ratios * Rs, axis=1)
            return np.mean(pdis)
        try:
            ps = self._target_dist.pdf(xs)
            qs = self._sample_dist.pdf(xs)
        except Exception:
            rs = []
            for x in xs:
                p = self._target_dist.pdf([x])[0]
                try:
                    q = self._sample_dist.pdf([x])[0]
                except Exception:
                    rs.append(1e10)
                else:
                    rs.append(p / q)
            rs = np.array(rs)
        else:
            rs = ps / qs
        wxs = rs * fs
        if self._weighted:
            return np.dot(rs, fs) / np.sum(rs)
        mean = np.mean(wxs)

        return mean


def optimize_lds(lds_dist):
    """CEM for optimizing parameters of LDS policy."""
    n_itr = 50
    n_evals = 20
    n_elite = 20
    n_samples = 100
    init_K = lds_dist._K
    print(init_K)
    k_shape = init_K.shape
    mean = init_K.flatten()
    cov = np.eye(np.size(mean))
    for itr in range(n_itr):

        # Evaluate initial distribtuion
        _, fs = lds_dist.sample(n=n_evals)
        print('Itr %d: %f (%f)' % (itr, np.mean(fs), np.std(fs)))

        # 1. Generate samples
        pop = np.random.multivariate_normal(mean, cov, size=n_samples)
        values = []
        for ind in pop:
            # Set vars from ind
            lds_dist._K = ind.reshape(k_shape)
            _, fs = lds_dist.sample(n=n_evals)
            values.append(np.mean(fs))

        idx = np.argsort(np.array(values))
        elite = pop[idx[-n_elite:]]
        mean = np.mean(elite, axis=0)
        cov = np.cov(elite.T)
        lds_dist._K = mean.reshape(k_shape)

    # Final Eval
    _, fs = lds_dist.sample(n=n_evals)
    print('Itr %d: %f (%f)' % (n_itr, np.mean(fs), np.std(fs)))
    print(lds_dist._K)


def main():  # noqa

    from lds import LDS
    np.random.seed(0)
    dimension = 2
    deg = 2
    true_degree = list(range(deg + 1))
    in_dim = len(true_degree) * 2 * dimension
    # k = np.random.normal(0, 0.1, size=(in_dim, dimension))
    k = np.ones((in_dim, dimension)) * 0.1
    # target distribution
    p = LDS(dimension, k, 0.001, polynomial_degrees=true_degree)
    optimize_lds(p)


if __name__ == '__main__':
    main()
