"""Implements different OPE methods."""
from __future__ import print_function
from __future__ import division

import numpy as np


class Estimator(object):

    def __init__(self):
        pass

    def estimate(self, paths):
        pass


class ISEstimate(Estimator):

    def __init__(self, pie, pib, weighted=False):
        self.pie = pie
        self.pib = pib
        self.weighted = weighted

    def _get_weights(self, paths):
        ws = np.ones(len(paths))
        for i, path in enumerate(paths):
            for ob, act in zip(path['obs'], path['acts']):
                p = self.pie[ob, act]
                q = self.pib[ob, act]
                ws[i] *= p / q
        return ws

    def estimate(self, paths):
        ws = self._get_weights(paths)
        Gs = np.array([sum(path['rews']) for path in paths])
        if self.weighted:
            ws /= np.sum(ws)
            return Gs.dot(ws), None

        wGs = np.multiply(Gs, ws)
        return np.mean(wGs), np.std(wGs) ** 2


class RISEstimate(ISEstimate):
    """Sub-class of IS that estimates behavior policy for the weights."""

    def __init__(self, pie, n_actions, horizon, n=None, weighted=False):  # noqa
        super(RISEstimate, self).__init__(pie, None, weighted=weighted)
        self.n_actions = n_actions
        self.L = horizon
        self.n = self.L
        if n is not None:
            self.n = n
        print('n=%d' % self.n)
        self.sa_counts = {}
        self.s_counts = {}
        self.n_paths = 0

    def _get_weights(self, paths):

        probs = {}
        # curr = []
        self.sa_counts[tuple([])] = len(paths)
        self.s_counts[tuple([])] = len(paths)
        probs[tuple([])] = len(paths)
        assert self.n_paths < len(paths), 'This should not happen.'
        for path in paths[self.n_paths:]:
            # update counts with new paths
            long_ob = []
            for ob, act in zip(path['obs'], path['acts']):
                long_ob.append(ob)
                if tuple(long_ob) not in self.s_counts:
                    self.s_counts[tuple(long_ob)] = 0.0
                    self.sa_counts[tuple(long_ob)] = np.zeros(self.n_actions)
                self.s_counts[tuple(long_ob)] += 1
                self.sa_counts[tuple(long_ob)][act] += 1
                long_ob.append(act)
                if len(long_ob) >= self.n * 2:
                    # Trim first obs and action
                    long_ob = long_ob[2:]
        self.n_paths = len(paths)

        for state in self.s_counts:
            probs[state] = self.sa_counts[state] / self.s_counts[state]

        ws = np.ones(len(paths))
        for i, path in enumerate(paths):
            long_ob = []
            for ob, act in zip(path['obs'], path['acts']):
                long_ob.append(ob)
                p = self.pie[ob, act]
                q = probs[tuple(long_ob)][act]

                ws[i] *= p / q
                long_ob.append(act)
                if len(long_ob) >= self.n * 2:
                    # Trim first obs and action
                    long_ob = long_ob[2:]
        return ws


class REGEstimate(ISEstimate):
    """The REG estimator from Li et al. (2015)."""

    def __init__(self, pie, mdp):  # noqa
        super(REGEstimate, self).__init__(pie, None)
        self.mdp = mdp
        self.rews = {}
        self.counts = {}
        self.probs = {}
        self.n_paths = 0

    def estimate(self, paths):

        Gs = np.array([sum(path['rews']) for path in paths[self.n_paths:]])

        for path, G in zip(paths[self.n_paths:], Gs):
            traj = []
            p = 1
            for t, (ob, act) in enumerate(zip(path['obs'], path['acts'])):
                traj.extend([ob, act])
                p *= self.pie[ob, act]
                if t < self.mdp.L - 1:
                    p *= self.mdp.transitions[ob, act, path['obs'][t + 1]]
            traj = tuple(traj)
            if traj not in self.counts:
                self.rews[traj] = 0
                self.counts[traj] = 0
            self.rews[traj] += G
            self.counts[traj] += 1
            self.probs[traj] = p
        self.n_paths = len(paths)
        estimate = 0

        for traj in self.rews:
            r = self.rews[traj] / float(self.counts[traj])
            estimate += self.probs[traj] * r
        return estimate, 0
