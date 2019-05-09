import numpy as np


def p_basis(x, orders=[1]):
    y = np.concatenate([np.power(x, o) for o in orders], axis=0)
    return y.flatten()


class LDS(object):
    """
    Describes trajectory distribution of agent moving in a plane.

    State given by x in R^4, action given by u in R^2.
    action = k * x + N(0, std)
    state = A * x + B * u + N(0, 1)
    """
    def __init__(self,
                 K,
                 sigma,
                 polynomial_degrees=[1],
                 L=20):

        # Environment constants
        dimension = 2
        A = np.eye(2 * dimension)
        B = np.vstack([0.5 * np.eye(dimension), np.eye(dimension)])

        def step(x, u):
            return A.dot(x) + B.dot(u)
        self._step = step
        self._horizon = L

        # goal is point (5, 5)
        goal = np.zeros(2 * dimension)
        for i in range(dimension):
            goal[2 * i] = 5

        def cost(x, u):
            return -1 * np.linalg.norm(x - goal)
        self._cost = cost

        # Agent policy
        self._K = np.array(K, copy=True)
        self._sigma = sigma * np.ones(dimension, dtype=float)

        base_in_dim = 2 * dimension
        if polynomial_degrees:
            in_dim = len(polynomial_degrees) * 2 * dimension
        else:
            in_dim = 1

        def input_fn(x):
            # if polynomial_degree == 1:
            #     return x
            rows = []
            for row in x.reshape(-1, base_in_dim):
                rows.append(p_basis(row, orders=polynomial_degrees))
            y = np.concatenate(rows, axis=0).reshape(-1, in_dim)
            return y

        # Class private member variables
        self._dimension = dimension
        self._in_dim = in_dim
        self._input_fn = input_fn

    def sample(self, n=1, policy=None):
        paths = []
        gs = np.zeros(n)
        for i in range(n):
            state = np.zeros(2 * self._dimension)
            x = np.zeros((self._horizon, 2 * self._dimension))
            u = np.zeros((self._horizon, self._dimension))
            r = np.zeros(self._horizon)
            pi_noise = np.random.normal(0, self._sigma, (self._horizon, self._dimension))
            noise = np.random.normal(0, 0.05, (self._horizon, self._dimension * 2))
            for t in range(self._horizon):
                control = self.mean(self._input_fn(state)) + pi_noise[t]
                x[t] = state
                u[t] = control
                state = self._step(state, control) + noise[t]
                state = np.clip(state, -10, 10)
                r[t] = self._cost(state, u)
            g = np.sum(r)
            gs[i] = g
            paths.append({'x': x, 'u': u, 'r': r})
        return paths, gs

    def expected_value(self):
        _, fs = self.sample(n=100000)
        print('True Eval CI %f' % (np.std(fs) * 1.96 * 0.01))
        return np.mean(fs)

    def mean(self, x):
        mean = x.dot(self._K)
        return mean.flatten()

    def mle_fit(self, paths):
        """
        Fit _K with ordinary least squares. Fit _sigma with residuals.

        paths: dict, must include keys 'x' and 'u' which are numpy arrays.
        """
        xs = np.concatenate([path['x'] for path in paths])
        xs = self._input_fn(xs)
        us = np.concatenate([path['u'] for path in paths])
        self._K = np.linalg.pinv(xs.T.dot(xs)).dot(xs.T.dot(us))
        self._sigma = np.sqrt(np.mean(np.square(us - xs.dot(self._K)), axis=0))

    def _action_log_likelihood(self, x, u):
        y = self._input_fn(x)
        mean = y.dot(self._K)
        log_std = np.log(self._sigma)
        zs = (u - mean) / np.exp(log_std)
        lh = - np.sum(log_std) - \
            0.5 * np.sum(np.square(zs)) - \
            0.5 * self._dimension * np.log(2 * np.pi)
        return lh

    def _pdf_u(self, x, u):
        lh = self._action_log_likelihood(x, u)
        return np.exp(lh)

    def _pdf(self, path):
        xs = path['x']
        us = path['u']
        ll = [self._action_log_likelihood(x, u) for x, u in zip(xs, us)]
        return np.exp(np.sum(ll))

    def pdf(self, paths):
        return np.array([self._pdf(path) for path in paths])

    def nlh(self, paths):
        xs = np.concatenate([path['x'] for path in paths])
        us = np.concatenate([path['u'] for path in paths])
        ll = [self._action_log_likelihood(x, u) for x, u in zip(xs, us)]
        return -1 * np.sum(ll)

    def ll_path(self, path):
        return [self._action_log_likelihood(x, u) for x, u in zip(path['x'],
                                                                  path['u'])]

    def pdf_per_decision(self, paths):
        return np.array(
            [np.exp(np.cumsum(self.ll_path(path))) for path in paths])

