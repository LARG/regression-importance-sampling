"""Defines SinglePath MDP and utils."""
from __future__ import print_function
from __future__ import division

import numpy as np


def sample(mdp, pi):
    """Generate a trajectory from mdp with pi."""
    done = False
    obs = mdp.reset()
    G = 0
    path = {}
    path['obs'] = []
    path['acts'] = []
    path['rews'] = []
    while not done:
        action = np.random.choice(2, p=pi[obs])
        path['obs'].append(obs)
        path['acts'].append(action)
        obs, rew, done = mdp.step(action)
        G += rew
        path['rews'].append(rew)
    return path, G


def evaluate(mdp, pi):
    """Run value iteration to evaluate policy exactly."""
    P = mdp.transitions
    L = mdp.L
    S = mdp.n_states
    A = mdp.n_actions
    V = np.zeros((L + 1, S + 1))
    for t in reversed(range(L)):
        for s in range(S):
            for a in range(A):
                V[t, s] += pi[s, a] * mdp.rewards[s, a]
                if t + 1 < L:
                    V[t, s] += pi[s, a] * P[s, a].dot(V[t + 1])
    return V[0, mdp.init_state]


class MDP(object):

    def __init__(self, num_states, num_actions, horizon):
        self.n_states = num_states
        self.n_actions = num_actions
        self.L = horizon

    def reset(self):
        pass

    def step(self, action):
        states = np.arange(self.n_states + 1)
        n_state = np.random.choice(states,
                                   p=self.transitions[self.state, action])
        rew = self.rewards[self.state, action]
        self.state = n_state
        return n_state, rew, False

    def get_policy(self, num):
        pass


class SinglePathMDP(MDP):

    def __init__(self, num_actions, horizon, stochastic=True):
        super(SinglePathMDP, self).__init__(horizon, num_actions, horizon)
        self.rewards = np.ones((horizon, num_actions)) * np.array([1, 0])
        self.state = 0
        self.t = 0
        self.init_state = 0
        self.transitions = np.zeros((horizon, num_actions, horizon + 1))
        p = np.zeros(num_actions)
        if stochastic:
            p = np.arange(num_actions) / float(num_actions)
        for t in range(horizon):
            for a in range(num_actions):
                self.transitions[t, a, t + 1] = 1.0 - p[a]
                self.transitions[t, a, t] = p[a]

    def step(self, action):
        s, rew, _ = super(SinglePathMDP, self).step(action)
        self.t += 1
        done = self.t == self.L
        return s, rew, done

    def reset(self):
        self.t = 0
        self.state = self.init_state
        return self.state

    def get_policy(self, num):
        p = 0.5
        if num == 0:
            p = 0.4
        elif num == 1:
            p = 0.6
        q = (1 - p) / (self.n_actions - 1)
        mult = np.array([p] + [q for _ in range(self.n_actions - 1)])
        pi = np.ones((self.L, self.n_actions)) * mult
        return pi
