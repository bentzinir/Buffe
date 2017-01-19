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
            'lr': 1e-4,
            'batch_size': batch_size
        }

        # set all the necessary weights and biases according to the forward model structure
        self.weights = OrderedDict()

        self.weights.update(self.gru_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], "gru1"))

        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'decoder1'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['output_dim'], 'decoder2'))
        # self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['small_encoding_dim']*self.arch_params['output_dim'], 'multiheaded1'))
        # self.weights.update(self.tensor_linear_variables(self.arch_params['small_encoding_dim'],
        #                                           self.arch_params['output_dim'], 1, 'multiheaded2'))

        self.weights.update(self.linear_variables(state_size, self.arch_params['encoding_dim'], 'encoder1_state'))
        self.weights.update(self.linear_variables(action_size, self.arch_params['encoding_dim'], 'encoder1_action'))

        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2_state'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder2_action'))

        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder3'))
        self.weights.update(self.linear_variables(self.arch_params['encoding_dim'], self.arch_params['encoding_dim'], 'encoder4'))

        #self.weights.update(self.bn_variables([1, self.arch_params['encoding_dim']], 'bn1'))
        #self.weights.update(self.bn_variables([1, self.arch_params['encoding_dim']], 'bn2'))

        self.states_normalizer = []
        self.actions_normalizer = []
        self.states_min = []

    def gru_variables(self, hidden_size, input_size, name):
        weights = OrderedDict()
        weights[name+'_Wxr'] = self.weight_variable([input_size, hidden_size])
        weights[name+'_Wxz'] = self.weight_variable([input_size, hidden_size])
        weights[name+'_Wxh'] = self.weight_variable([input_size, hidden_size])
        weights[name+'_Whr'] = self.weight_variable([hidden_size, hidden_size])
        weights[name+'_Whz'] = self.weight_variable([hidden_size, hidden_size])
        weights[name+'_Whh'] = self.weight_variable([hidden_size, hidden_size])
        weights[name+'_br'] = self.bias_variable([1, hidden_size])
        weights[name+'_bz'] = self.bias_variable([1, hidden_size])
        weights[name+'_bh'] = self.bias_variable([1, hidden_size])
        return weights

    def bn_variables(self, size, name):
        weights = OrderedDict()
        weights[name+'_mean'] = tf.Variable(tf.constant(0.0, shape=size))
        weights[name +'_variance'] = tf.Variable(tf.constant(1.0, shape=size))
        weights[name + '_offset'] = tf.Variable(tf.constant(0.0, shape=size))
        weights[name + '_scale'] = tf.Variable(tf.constant(1.0, shape=size))
        return weights

    def tensor_linear_variables(self, input_width, input_depth, output_width, name):
        weights = OrderedDict()
        self.weights[name+'_weights'] = self.weight_variable([input_depth, input_width, output_width])
        self.weights[name+'_biases'] = self.bias_variable([input_depth, 1, output_width])
        return weights

    def linear_variables(self, input_size, output_size, name):
        weights = OrderedDict()
        self.weights[name+'_weights'] = self.weight_variable([input_size, output_size])
        self.weights[name+'_biases'] = self.bias_variable([1, output_size])
        return weights

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def kl_divergence(self, p, p_hat):
        # returns KL(p||p_hat) - the kl divergence between two Bernoulli probability vectors p and p_hat
        return p*(tf.log(p) - tf.log(p_hat)) + (1-p)*(tf.log(1-p)-tf.log(1-p_hat))

    def gru_layer(self, input, hidden, weights, name):
        x, h_ = input, hidden
        r = tf.sigmoid(tf.matmul(x, weights[name+'_Wxr']) + tf.matmul(h_, weights[name+'_Whr']) + weights[name+'_br'])
        z = tf.sigmoid(tf.matmul(x, weights[name+'_Wxz']) + tf.matmul(h_, weights[name+'_Whz']) + weights[name+'_bz'])

        h_hat = tf.tanh(
            tf.matmul(x, weights[name+'_Wxh']) + tf.matmul(tf.mul(r, h_), weights[name+'_Whh']) + weights[name+'_bh'])

        output = tf.mul((1 - z), h_hat) + tf.mul(z, h_)

        return output

    def encode(self, input):
        state = tf.cast(input[0], tf.float32)
        action = tf.cast(input[1], tf.float32)
        gru_state = tf.cast(input[2], tf.float32)

        # returns an encoder
        state_embedder1 = tf.nn.relu(tf.matmul(state, self.weights["encoder1_state_weights"]) + self.weights["encoder1_state_biases"])
        gru_state = self.gru_layer(state_embedder1, gru_state, self.weights, 'gru1')
        state_embedder2 = tf.sigmoid(tf.matmul(gru_state, self.weights["encoder2_state_weights"]) + self.weights["encoder2_state_biases"])

        action_embedder1 = tf.nn.relu(tf.matmul(action, self.weights["encoder1_action_weights"]) + self.weights["encoder1_action_biases"])
        action_embedder2 = tf.sigmoid(tf.matmul(action_embedder1, self.weights["encoder2_action_weights"]) + self.weights["encoder2_action_biases"])

        output = self.merger(state_embedder2, action_embedder2)

        hidden = tf.matmul(output, self.weights["encoder3_weights"]) + self.weights["encoder3_biases"]
        #bn = tf.nn.batch_normalization(hidden, mean=self.weights["bn1_mean"], variance=self.weights["bn1_variance"],
        #                               offset=self.weights["bn1_offset"], scale=self.weights["bn1_scale"], variance_epsilon=0.001)
        hidden_relu = tf.nn.relu(hidden)
        output = tf.nn.relu(tf.matmul(hidden_relu, self.weights["encoder4_weights"]) + self.weights["encoder4_biases"])


        gru_state = tf.cast(gru_state, tf.float32)
        return output, gru_state

    def decode(self, input):
        # returns a decoder
        hidden = tf.matmul(input, self.weights["decoder1_weights"]) + self.weights["decoder1_biases"]
        hidden_relu = tf.nn.relu(hidden)

        # output is encoding_size x 1 x small_encoding_size
        # multiheaded_hidden = tf.matmul(input, self.weights["multiheaded1_weights"]) + self.weights["multiheaded1_biases"]
        # multiheaded_hidden = tf.reshape(multiheaded_hidden, [-1, self.arch_params['output_dim'], 1, self.arch_params['small_encoding_dim']])
        # multiheaded_hidden = tf.nn.relu(multiheaded_hidden)
        #
        # h = tf.scan(lambda a,x: tf.batch_matmul(x, self.weights["multiheaded2_weights"]), multiheaded_hidden,
        #            initializer=tf.Variable(tf.constant(0.0, shape=[self.arch_params['output_dim'],1,1])))
        # multiheaded_output = h + self.weights["multiheaded2_biases"]
        # output1 = tf.reshape(multiheaded_output, [-1, self.arch_params['output_dim']])

        output1 = tf.matmul(hidden_relu, self.weights["decoder2_weights"]) + self.weights["decoder2_biases"]
        output = output1
        return output

    def forward(self, input):
        # run a forward pass
        encoding, gru_state = self.encode(input)
        output = self.decode(encoding)
        sparsity_loss = []
        #sparsity_loss = tf.reduce_sum(self.kl_divergence(self.sparsity_params['rho'], encoding))
        return output, sparsity_loss, gru_state

    def calculate_loss(self, output, target, sparsity_loss):
        target = tf.cast(target, dtype=tf.float32)
        l2_loss = tf.nn.l2_loss(self.states_normalizer*(output - target)) / float(self.training_params['batch_size'])
        return l2_loss, l2_loss + self.sparsity_params['beta']*sparsity_loss

    def get_model(self):
        input_state = tf.placeholder(tf.float32, shape=[None, self.state_size], name='input_state')
        input_action = tf.placeholder(tf.float32, shape=[None, self.action_size], name='input_action')
        gru_state = tf.placeholder(tf.float32, shape=[None, self.arch_params['encoding_dim']], name='gru_state')
        input = [input_state, input_action, gru_state]
        output, sparsity_loss, new_gru_state = self.forward(input)
        target = tf.placeholder(tf.float32, shape=[None, self.arch_params['output_dim']], name='target')
        l2_loss, loss = self.calculate_loss(output, target, sparsity_loss)
        return input, target, output, loss, l2_loss, new_gru_state

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

    def pretrain(self, opt, lr, batch_size, num_iterations, expert_er_path):
        er_expert = common.load_er(
            fname=expert_er_path,
            batch_size=batch_size,
            history_length=100,
            traj_length=2)

        self.states_normalizer = er_expert.states_max - er_expert.states_min
        self.actions_normalizer = er_expert.actions_max - er_expert.actions_min
        #self.states_normalizer = er_expert.states_std
        #self.actions_normalizer = er_expert.actions_std
        self.states_normalizer[self.states_normalizer < 0.0001] = 1
        self.actions_normalizer[self.actions_normalizer < 0.0001] = 1
        self.states_min=er_expert.states_min

        # get placeholders
        input_ph, target_ph, output, loss, l2_loss, new_gru_state = self.get_model()
        train_op = opt(learning_rate=lr).minimize(l2_loss)

        train_losses = []
        last_train_losses = []
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for i in range(num_iterations):

                fetches = [train_op, output, l2_loss, loss, new_gru_state]

                # get a trajectory from the train / test set and preprocess it
                trajectory = er_expert.sample_trajectory(self.num_steps + 1)
                trajectory_states = (trajectory[0] - er_expert.states_min) / self.states_normalizer
                trajectory_actions = (trajectory[1] - er_expert.actions_min) / self.actions_normalizer

                # set inputs and targets
                s = np.ones((1, self.arch_params['encoding_dim']))

                #o = trajectory_states[:batch_size, 0, :]
                target = []
                for step in range(self.num_steps):
                    input = [trajectory_states[:batch_size, step, :], trajectory_actions[:batch_size, step, :]]
                    target = np.squeeze(trajectory_states[:batch_size, step+1, :])

                    _, o, l2, l, s = sess.run(fetches,
                                           feed_dict={input_ph[0]: input[0], input_ph[1]: input[1], input_ph[2]: s, target_ph: target})

                if i % 50 == 0:
                    print("iteration " + str(i) + " l2 loss = " + str(l2))
                    print((self.states_normalizer*o[0])[:30])
                    print((self.states_normalizer*target[0])[:30])
                    print("***********************")

                last_train_losses += [l2]
                if len(last_train_losses) > 50:
                    del last_train_losses[0]

                train_losses += [sum(last_train_losses) / float(len(last_train_losses))]
