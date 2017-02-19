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
        self.dropout_keep = dropout_keep
        self.lstm = lstm
        self.arch_params = {
            'input_dim': lstm and (state_size + action_size) or (state_size + action_size),
            'encoding_dim': encoding_size,
            'small_encoding_dim': 5,
            'output_dim': state_size
        }
        self.sparsity_params = {
            'rho': tf.constant(rho),
            'beta': tf.constant(beta)
        }
        self.training_params = {
            'lr': 1e-2,
            'batch_size': batch_size
        }

        # set all the necessary weights and biases according to the forward model structure
        self.weights = OrderedDict()

        self.weights.update(self.linear_variables(state_size + action_size, self.arch_params['encoding_dim'], 'encoder1'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['output_dim'], 'encoder3'))
        
        self.states_normalizer = []
        self.actions_normalizer = []
        self.states_min = []

    def linear_variables(self, input_size, output_size, name):
        weights = OrderedDict()
        self.weights[name+'_weights'] = self.weight_variable([input_size, output_size])
        self.weights[name+'_biases'] = self.weight_variable([1, output_size])
        return weights

    def weight_variable(self, shape):
        initial = tf.random_normal(shape, stddev=0.01, dtype=tf.float32)
        return tf.Variable(initial)

    def encode(self, input):
        state = tf.cast(input[0], tf.float32)
        action = tf.cast(input[1], tf.float32)
        gru_state = tf.cast(input[2], tf.float32)

        concat = tf.concat(concat_dim=1, values=[state, action], name='input')

        h0 = tf.nn.relu(tf.matmul(concat, self.weights["encoder1_weights"]) + self.weights["encoder1_biases"])

        h1 = tf.nn.relu(tf.matmul(h0, self.weights["encoder2_weights"]) + self.weights["encoder2_biases"])

        h1_do = tf.nn.dropout(h1, self.dropout_keep)

        delta = tf.matmul(h1_do, self.weights["encoder3_weights"]) + self.weights["encoder3_biases"]
        
        previous_state = tf.stop_gradient(state)

        output = previous_state + delta
        
        return output, gru_state

    def forward(self, input):
        print('Forward Model Vanilla')
        # run a forward pass
        output, gru_state = self.encode(input)
        sparsity_loss = []
        return output, sparsity_loss, gru_state

    def backward(self, loss):

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.training_params['lr'])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights.values())
        mean_abs_grad, mean_abs_w = common.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, mean_abs_grad, mean_abs_w

    def train(self, objective):
        self.loss = objective
        self.minimize, self.mean_abs_grad, self.mean_abs_w = self.backward(self.loss)
        self.loss_summary = tf.scalar_summary('loss_t', objective)

