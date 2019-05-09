import numpy as np
import tensorflow as tf
import tflearn
from gym import spaces


def _get_space_size(space):
    if hasattr(space, 'n'):
        return space.n
    elif hasattr(space, 'low'):
        return np.size(space.low, axis=0)


def _get_space_dims(space):
    if hasattr(space, 'n'):
        return 1
    elif hasattr(space, 'low'):
        return space.low.shape


def get_policy_class(policy_str):
    if policy_str == 'boltzmann':
        return ContinuousStateBoltzmannPolicy
    if policy_str in ['gaussian', 'Gaussian']:
        return GaussianPolicy


class Policy(object):

    def __init__(self, observation_space, action_space):
        pass

    def get_action(self, observation):
        pass

    def pdf(self, observation, action):
        pass

    def grad_log_policy(self, observation, action):
        pass


class RandomPolicy(Policy):

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Box):
            self.r = self.action_space.high - self.action_space.low
        else:
            self.r = self.action_space.n

    def get_action(self, observation):
        return self.action_space.sample(), 1. / self.r

    def pdf(self, observation, action):
        return 1. / self.r

    def grad_log_policy(self, obs, act):
        return 0.


class NeuralNetworkPolicy(Policy):
    """
    Policy that computes action selection distribution with a neural net
    function of the given observation. Base class for GaussianPolicy and
    ContinuousStateBoltzmannPolicy.
    """

    def __init__(self, observation_space, action_space, scope=None,
                 learning_rate=1e-04,
                 hidden_sizes=[],
                 act_fn=tf.nn.relu,
                 filter_obs=False,
                 seed=None,
                 entropy_coeff=None,
                 weight_decay=0.0):
        """
        observation_space:
        n_logits:
        scope:
        learning_rate:
        hidden_sizes:
        train_type:
        act_fn:
        filter_obs:
        seed:
        entropy_coeff:
        """
        obs_dim = _get_space_size(observation_space)
        n_logits = _get_space_size(action_space)
        self.graph = tf.Graph()
        scope = '' if scope is None else scope
        self.entropy_coeff = entropy_coeff
        self.learning_rate = learning_rate
        if seed is not None:
            np.random.seed(seed)

        with self.graph.as_default():

            with tf.variable_scope(scope):

                self.params = []
                self.obs_input = tflearn.input_data(shape=[None, obs_dim],
                                                    name='obs_input')
                self.obs_dim = tuple([-1, obs_dim])

                if filter_obs:
                    with tf.variable_scope('obfilter'):

                        self.rms_sum = tf.get_variable(
                            dtype=tf.float64,
                            shape=obs_dim,
                            initializer=tf.constant_initializer(1.0),
                            name='runningsum', trainable=False)
                        self.rms_sumsq = tf.get_variable(
                            dtype=tf.float64,
                            shape=obs_dim,
                            initializer=tf.constant_initializer(1.0),
                            name='runningsumsq', trainable=False)
                        self.rms_count = tf.get_variable(
                            dtype=tf.float64,
                            shape=(),
                            initializer=tf.constant_initializer(1.0),
                            name='count', trainable=False)
                        mean = tf.to_float(self.rms_sum / self.rms_count)
                        var = tf.to_float(self.rms_sumsq / self.rms_count)
                        var = var - tf.square(mean)
                        var = tf.maximum(var, 1e-2)
                        std = tf.sqrt(var)
                        self.params.extend([self.rms_sum, self.rms_sumsq,
                                            self.rms_count])

                    prev = tf.clip_by_value((self.obs_input - mean) / std,
                                            -5.0, 5.0)
                else:
                    prev = self.obs_input

                init = tflearn.initializations.truncated_normal(seed=seed)
                for idx, size in enumerate(hidden_sizes):
                    prev = tflearn.fully_connected(prev, size,
                                                   name='hidden_layer%d' % idx,
                                                   activation=act_fn,
                                                   weights_init=init,
                                                   weight_decay=weight_decay)
                    self.params.extend([prev.W, prev.b])

                self.logits = tflearn.fully_connected(prev, n_logits,
                                                      name='logits',
                                                      weights_init=init,
                                                      weight_decay=weight_decay)
                self.params.extend([self.logits.W, self.logits.b])

    def initialize(self):
        self.session.run(self.init_op)

    def reinforce_update(self, observations, actions, advantages):
        feed_dict = {self.obs_input: observations,
                     self.adv_var: advantages}
        _, loss = self.session.run([self.train_step, self.loss],
                                   feed_dict=feed_dict)
        return loss

    def supervised_update(self, observations, actions):
        act_in = actions
        if hasattr(self, 'n_actions'):
            acts = np.zeros(shape=(len(actions), self.n_actions))
            for i, act in enumerate(actions):
                acts[i, act] = 1.
            act_in = acts
        feed_dict = {self.obs_input: observations,
                     self.act_in: act_in}
        _, loss = self.session.run([self.train_step, self.loss],
                                   feed_dict=feed_dict)
        return loss

    def eval_loss(self, observations, actions):
        act_in = actions
        if hasattr(self, 'n_actions'):
            acts = np.zeros(shape=(len(actions), self.n_actions))
            for i, act in enumerate(actions):
                acts[i, act] = 1.
            act_in = acts
        feed_dict = {self.obs_input: observations,
                     self.act_in: act_in}
        loss = self.session.run(self.loss, feed_dict=feed_dict)
        return loss

    def save_policy(self, modelpath):
        vs = {}
        for param in self.params:
            vs[param.name] = param
        saver = tf.train.Saver(vs)
        saver.save(self.session, modelpath)

    def load_policy(self, modelpath):
        print ('Loading %s' % modelpath)
        vs = {}
        for param in self.params:
            vs[param.name] = param
        saver = tf.train.Saver(vs)
        saver.restore(self.session, modelpath)

    def get_dist_vars(self, observation):
        pass

    def entropy(self, observations):
        feed_dict = {self.obs_input: observations}
        return self.session.run(self.avg_entropy, feed_dict=feed_dict)


class ContinuousStateBoltzmannPolicy(NeuralNetworkPolicy):

    def __init__(self, observation_space, action_space, scope=None,
                 learning_rate=1e-04,
                 hidden_sizes=[],
                 act_fn=tf.nn.relu,
                 filter_obs=False,
                 seed=None,
                 entropy_coeff=None,
                 weight_decay=0.0):
        self.n_actions = _get_space_size(action_space)
        scope = '' if scope is None else scope
        super(ContinuousStateBoltzmannPolicy,
              self).__init__(observation_space, action_space, scope=scope,
                             learning_rate=learning_rate,
                             hidden_sizes=hidden_sizes,
                             act_fn=act_fn,
                             filter_obs=filter_obs,
                             seed=seed,
                             entropy_coeff=entropy_coeff,
                             weight_decay=0.0)

        with self.graph.as_default():

            with tf.variable_scope(scope):

                self.action_probs = tf.nn.softmax(self.logits)
                log_probs = tf.log(self.action_probs)
                self.grad_log_prob = tf.gradients(log_probs, self.params)
                entropy = self.action_probs * tf.log(self.action_probs)
                entropy = tf.negative(tf.reduce_sum(entropy, axis=0))
                self.avg_entropy = tf.reduce_mean(entropy)

                # Loss is sum of neg log likelihoods with optional entropy
                # term. We also compute an avg_loss without the average
                # entropy
                self.act_in = tflearn.input_data(shape=[None,
                                                        self.n_actions],
                                                 name='Actions')
                cross_entropy = tf.negative(
                    self.act_in * tf.log(self.action_probs))
                self.loss = tf.reduce_sum(cross_entropy)
                if self.entropy_coeff not in [0.0, None]:
                    print('Coeff %f' % self.entropy_coeff)
                    self.loss -= self.entropy_coeff * tf.reduce_sum(entropy)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = optimizer.minimize(self.loss)

                self.init_op = tf.global_variables_initializer()

        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_op)

    def _get_action_probs(self, observation):
        probs = self.session.run(
            self.action_probs,
            feed_dict={self.obs_input: observation.reshape(self.obs_dim)})
        return probs[0]

    def get_action(self, observation):
        probs = self._get_action_probs(observation)
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs[action]

    def pdf(self, observation, action):
        probs = self._get_action_probs(observation)
        return probs[action]

    def grad_log_policy(self, observation, action, param=0):
        obs = observation.reshape(self.obs_dim)
        grad_log_p = self.session.run(self.grad_log_prob,
                                      feed_dict={self.obs_input: obs})
        grads = np.concatenate([glp.reshape(
            -1, self.n_actions) for glp in grad_log_p], axis=0)

        return grads[param][action]

    def reinforce_update(self, observations, actions, advantages):
        advs = np.zeros(shape=(len(actions), self.n_actions))
        for i, act in enumerate(actions):
            advs[i, act] = advantages[i]
        return super(ContinuousStateBoltzmannPolicy,
                     self).reinforce_update(observations, actions, advs)

    def get_dist_vars(self, observation):
        logits = self.session.run(
            self.logits,
            feed_dict={self.obs_input: observation.reshape(self.obs_dim)})
        return {'logits': logits}


class GaussianPolicy(NeuralNetworkPolicy):

    def __init__(self, observation_space, action_space, scope=None,
                 learning_rate=1e-04,
                 hidden_sizes=[],
                 train_type='supervised',
                 act_fn=tf.nn.relu,
                 filter_obs=False,
                 seed=None,
                 learn_std=True,
                 entropy_coeff=None,
                 weight_decay=0.0):
        self.action_dim = _get_space_size(action_space)
        scope = '' if scope is None else scope
        super(GaussianPolicy, self).__init__(observation_space, action_space,
                                             scope=scope,
                                             learning_rate=learning_rate,
                                             hidden_sizes=hidden_sizes,
                                             train_type=train_type,
                                             act_fn=act_fn,
                                             filter_obs=filter_obs,
                                             seed=seed,
                                             entropy_coeff=entropy_coeff,
                                             weight_decay=0.0)

        with self.graph.as_default():

            with tf.variable_scope(scope):

                self.mean = self.logits

                self.log_std = tf.get_variable(
                    'logstd', initializer=tf.zeros(self.action_dim),
                    trainable=learn_std)
                self.params.append(self.log_std)
                self.act_in = tflearn.input_data(shape=[None, self.action_dim],
                                                 name='Actions')

                zs = (self.act_in - self.mean) / tf.exp(self.log_std)
                self.log_likelihood = - tf.reduce_sum(self.log_std, axis=-1) - \
                    0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
                    0.5 * self.action_dim * np.log(2 * np.pi)
                self.grad_log_prob = tf.gradients(self.log_likelihood,
                                                  self.params)
                ent = tf.log(np.sqrt(2 * np.pi * np.e, dtype=np.float32))
                entropy = self.log_std + ent
                self.avg_entropy = tf.reduce_mean(entropy, axis=-1)
                if self.train_type == 'reinforce':
                    self.adv_var = tflearn.input_data(shape=[None, 1])
                    self.loss = tf.reduce_sum(tf.multiply(self.adv_var,
                                                          self.log_likelihood))
                    optimizer = tf.train.GradientDescentOptimizer(
                        self.learning_rate)
                    self.train_step = optimizer.minimize(
                        tf.negative(self.loss))
                elif self.train_type == 'supervised':
                    # Loss is sum of neg log likelihoods with optional entropy
                    # term. We also compute an avg_loss without the average
                    # entropy
                    self.loss = tf.reduce_sum(
                        tf.negative(self.log_likelihood))
                    # negloglikelihood = tf.negative(self.log_likelihood)
                    # self.loss = tf.reduce_sum(negloglikelihood)
                    if self.entropy_coeff not in [0.0, None]:
                        print('Coeff %f' % self.entropy_coeff)
                        self.loss -= self.entropy_coeff * tf.reduce_sum(entropy)
                    else:
                        print('No Entropy Regularization')
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.train_step = optimizer.minimize(self.loss)

                self.init_op = tf.global_variables_initializer()

        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_op)

    def get_action(self, observation, stochastic=True):
        mean, log_std = self.session.run(
            [self.mean, self.log_std],
            feed_dict={self.obs_input: observation.reshape(self.obs_dim)})
        if not stochastic:
            return mean.flatten(), 1.0
        rnd = np.random.normal(size=self.action_dim)
        # print(rnd[0:4], mean[0:4])
        action = rnd * np.exp(log_std) + mean
        return action.flatten(), self.pdf(observation, action)

    def pdf(self, observation, action):
        feed_dict = {self.obs_input: observation.reshape(self.obs_dim),
                     self.act_in: action.reshape((1, self.action_dim))}
        log_li = self.session.run(self.log_likelihood, feed_dict=feed_dict)
        prob = np.exp(log_li)
        return prob.flatten()[0]

    def grad_log_policy(self, observation, action, param=0):
        feed_dict = {self.obs_input: observation,
                     self.act_in: action}
        return self.session.run(self.grad_log_pi, feed_dict=feed_dict)

    def reinforce_update(self, observations, actions, advantages):
        feed_dict = {self.obs_input: observations,
                     self.act_in: actions,
                     self.adv_var: advantages.reshape(-1, 1)}
        _, loss = self.session.run([self.train_step, self.loss],
                                   feed_dict=feed_dict)
        return loss

    def get_dist_vars(self, observation):
        mean, log_std = self.session.run(
            [self.mean, self.log_std],
            feed_dict={self.obs_input: observation.reshape(self.obs_dim)})
        return {'mean': mean, 'log_std': log_std}


class MixturePolicy(Policy):

    def __init__(self, policies, weights):
        self.policies = policies
        self.weights = np.array(weights)
        assert np.sum(self.weights) == 1.0, 'Weights must sum to 1.'

    def get_action(self, observation):
        pol_ind = np.random.choice(len(self.policies), p=self.weights)
        return self.policies[pol_ind].get_action(observation)

    def pdf(self, observation, action):
        pdfs = np.array([pi.pdf(observation, action) for pi in self.policies])
        return self.weights.dot(pdfs)

    def grad_log_policy(self, obs, act):
        return 0.
