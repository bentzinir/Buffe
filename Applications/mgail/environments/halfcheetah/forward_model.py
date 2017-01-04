import tensorflow as tf
import numpy as np
import common
import matplotlib.pyplot as plt
from collections import OrderedDict

class ForwardModel(object):
    def __init__(self, state_size, action_size, rho=0.05, beta=0.3, encoding_size=50, batch_size=50, multi_layered_encoder=True, num_steps=1,
                 separate_encoders=True, merger=tf.mul, activation=tf.sigmoid, dropout_keep=0.5, lstm=False):
        self.state_size = state_size
        self.action_size = action_size
        self.multi_layered_encoder = multi_layered_encoder
        self.separate_encoders = separate_encoders
        self.merger = merger
        self.num_steps = num_steps
        self.activation = activation
        self.dropout_keep = 0.5
        self.lstm = lstm
        self.arch_params = {
            'input_dim': lstm and (state_size + action_size) or num_steps*(state_size + action_size),
            'encoding_dim': encoding_size,
            'output_dim': state_size
        }
        self.sparsity_params = {
            'rho': tf.constant(rho),
            'beta': tf.constant(beta)
        }
        self.training_params = {
            'lr': 1e-4,
            'batch_size': batch_size
        }

        # set all the necessary weights and biases according to the forward model structure
        self.weights = OrderedDict()
        self.biases = OrderedDict()

        self.biases['decoder1'] = self.bias_variable([1, self.arch_params['encoding_dim']])
        self.weights['decoder1'] = self.weight_variable([self.arch_params['encoding_dim'], self.arch_params['encoding_dim']])
        self.weights['decoder2'] = self.weight_variable([self.arch_params['encoding_dim'], self.arch_params['output_dim']])
        self.biases['decoder2'] = self.bias_variable([1, self.arch_params['output_dim']])

        self.weights['encoder_state'] = self.weight_variable([num_steps * state_size, self.arch_params['encoding_dim']])
        self.weights['encoder_action'] = self.weight_variable([num_steps * action_size, self.arch_params['encoding_dim']])
        self.biases['encoder_state'] = self.bias_variable([1, self.arch_params['encoding_dim']])
        self.biases['encoder_action'] = self.bias_variable([1, self.arch_params['encoding_dim']])

        self.weights['encoder2'] = self.weight_variable([self.arch_params['encoding_dim'], self.arch_params['encoding_dim']])
        self.weights['encoder3'] = self.weight_variable([self.arch_params['encoding_dim'], self.arch_params['encoding_dim']])
        self.biases['encoder2'] = self.bias_variable([1, self.arch_params['encoding_dim']])
        self.biases['encoder3'] = self.bias_variable([1, self.arch_params['encoding_dim']])

        self.states_normalizer = []
        self.actions_normalizer = []

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def kl_divergence(self, p, p_hat):
        # returns KL(p||p_hat) - the kl divergence between two Bernoulli probability vectors p and p_hat
        return p*(tf.log(p) - tf.log(p_hat)) + (1-p)*(tf.log(1-p)-tf.log(1-p_hat))

    def encode(self, input):
        # returns an encoder
        output_state = tf.sigmoid(tf.matmul(input[0], self.weights["encoder_state"]) + self.biases["encoder_state"])
        output_action = tf.sigmoid(tf.matmul(input[1], self.weights["encoder_action"]) + self.biases["encoder_action"])
        output = self.merger(output_state, output_action)

        hidden = tf.nn.relu(tf.matmul(output, self.weights["encoder2"]) + self.biases["encoder2"])
        output = tf.sigmoid(tf.matmul(hidden + output, self.weights["encoder3"]) + self.biases["encoder3"])

        return output

    def decode(self, input):
        # returns a decoder
        hidden = tf.nn.relu(tf.matmul(input, self.weights["decoder1"]) + self.biases["decoder1"])
        output1 = tf.matmul(hidden, self.weights["decoder2"]) + self.biases["decoder2"]
        output = output1
        return output

    def forward(self, input):
        # run a forward pass
        encoding = self.encode(input)
        output = self.decode(encoding)
        sparsity_loss = tf.reduce_sum(self.kl_divergence(self.sparsity_params['rho'], encoding))
        return output, sparsity_loss

    def calculate_loss(self, output, target, sparsity_loss):
        l2_loss = tf.nn.l2_loss((output - target)*self.states_normalizer) / float(self.training_params['batch_size'])
        return l2_loss, l2_loss + self.sparsity_params['beta']*sparsity_loss

    def get_model(self):
        input_state = tf.placeholder(tf.float32, shape=[None, self.num_steps*self.state_size], name='input_state')
        input_action = tf.placeholder(tf.float32, shape=[None, self.num_steps*self.action_size], name='input_action')
        input = [input_state, input_action]
        output, sparsity_loss = self.forward(input)
        target = tf.placeholder(tf.float32, shape=[None, self.arch_params['output_dim']], name='target')
        l2_loss, loss = self.calculate_loss(output, target, sparsity_loss)
        return input, target, output, loss, l2_loss

    def backward(self, loss):

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.training_params['lr'])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values() + self.biases.values())
        mean_abs_grad, mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, mean_abs_grad, mean_abs_w

    def train(self, objective):
        self.loss = objective
        self.minimize, self.mean_abs_grad, self.mean_abs_w = self.backward(self.loss)
        self.loss_summary = tf.scalar_summary('loss_t', objective)

    def pretrain(self, opt, lr, batch_size, num_iterations, expert_er_path):
        er_expert = common.load_er(
            fname=expert_er_path,
            batch_size=batch_size,
            history_length=100,
            traj_length=2)

        self.states_normalizer = er_expert.states_max - er_expert.states_min
        self.actions_normalizer = er_expert.actions_max - er_expert.actions_min
        self.states_normalizer = er_expert.states_std
        self.actions_normalizer = er_expert.actions_std

        input_ph, target_ph, output, loss, l2_loss = self.get_model()
        train_op = opt(learning_rate=lr).minimize(loss)

        train_losses = []
        last_train_losses = []
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for i in range(num_iterations):

                fetches = [train_op, output, l2_loss, loss]

                # get a trajectory from the train / test set and pre-process it
                trajectory = er_expert.sample_trajectory(self.num_steps + 1)
                trajectory_states = trajectory[0] / self.states_normalizer
                trajectory_actions = trajectory[1] / self.actions_normalizer

                # set inputs and targets
                input = [trajectory_states[:batch_size, 0, :], trajectory_actions[:batch_size, 0, :]]
                for step in range(1, self.num_steps):
                    input = [np.concatenate((input[0], trajectory_states[:batch_size, step, :]), axis=-1),
                             np.concatenate((input[1], trajectory_actions[:batch_size, step, :]), axis=-1)]
                target = np.squeeze(trajectory_states[:batch_size, self.num_steps, :])

                _, o, l2, l = sess.run(fetches,
                                       feed_dict={input_ph[0]: input[0], input_ph[1]: input[1], target_ph: target})

                if i % 50 == 0:
                    print("iteration " + str(i) + " l2 loss = " + str(l2))

                last_train_losses += [l2]
                if len(last_train_losses) > 50:
                    del last_train_losses[0]

                train_losses += [sum(last_train_losses) / float(len(last_train_losses))]