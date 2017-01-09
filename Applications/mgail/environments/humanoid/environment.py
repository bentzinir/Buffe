import tensorflow as tf
import numpy as np
import gym

class ENVIRONMENT(object):

    def __init__(self, run_dir):

        self.name = 'Humanoid-v1'

        self.gym = gym.make(self.name)

        self._connect()

        self._train_params()

        self.run_dir = run_dir

    def _step(self, a):
        a = np.squeeze(a)
        self.t += 1
        state_ = self.state
        self.state, self.reward, self.done, self.info = self.gym.step(a)

        e = np.sqrt(((state_ - self.state) ** 2).sum())

        # if np.any(self.state[self.inactive_state_features] != 0):
        #     # print 'Not stable'
        #     self.done = True

        # if e < 0.05:
        #     # print 'Freeze'
        #     self.done = True
        return self.state.astype(np.float32), self.reward.astype(np.float32), self.done

    def step(self, a, mode):
        if mode == 'tensorflow':
            state, reward, done = tf.py_func(self._step, inp=[a], Tout=[tf.float32, tf.float32, tf.bool], name='env_step_func')
            # DEBUG: flatten state. not sure if correctly
            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            state, reward, done = self._step(a)
        info = 0.
        return state, reward, done, info

    def reset(self):
        self.t = 0
        self.state = self.gym.reset()
        return self.state

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

    def render(self):
        self.gym.render()

    def _connect(self):
        self.state_size = self.gym.observation_space.shape[0]
        self.action_size = self.gym.action_space.shape[0]
        self.action_space = np.asarray([None]*self.action_size)

    def _train_params(self):
        self.trained_model = None  #'/home/nir/work/git/Buffe/Applications/mgail/environments/humanoid/snapshots/2017-01-08-21-08-403000.sn'
        self.train_mode = True
        self.train_flags = [0, 1, 1]  # [autoencoder, transition, discriminator/policy]
        self.expert_data = 'expert_data/er-2017-01-03-14-50-240T-sorted.bin'
        self.disc_as_classifier = True
        self.pre_load_buffer = True
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.kill_itr = self.n_train_iters
        self.reward_kill_th = -1
        self.test_interval = 1000
        self.n_steps_test = 1000
        self.vis_flag = False
        self.save_models = False
        self.config_dir = None
        self.continuous_actions = True
        self.success_th = 4500
        self.weight_decay = 1e-7

        # Main parameters to play with:
        self.er_agent_size = 50000
        self.reset_itrvl = 10000
        self.n_reset_iters = 10000
        self.model_identification_time = 50000
        self.prep_time = 10000
        self.collect_experience_interval = 5
        self.n_steps_train = 2
        self.discr_policy_itrvl = 100
        self.K_T = 2
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 70
        self.policy_al_w = 1e-2
        self.policy_tr_w = 5e-3
        self.policy_accum_steps = 3
        self.total_trans_err_allowed = 100000# 1e-0
        # self.trans_loss_th = 50
        self.max_trans_iters = 2
        self.temp = 1.
        self.cost_sensitive_weight = 1.1

        self.t_size = [500, 300, 200]
        self.d_size = [800, 400]
        self.p_size = [600, 300]

        self.t_lr = 0.005
        self.d_lr = 0.001
        self.p_lr = 0.0001

        self.w_std = 0.15

        self.noise_intensity = 6.
        self.do_keep_prob = 0.85

        self.smooth_over_steps = 50
        self.fm_lr = 1e-2
        self.fm_rho = 0.1
        self.fm_beta = 0.01
        self.fm_encoding_size = 400
        self.fm_batch_size = 300
        self.fm_multi_layered_encoder = True
        self.fm_opt = tf.train.AdamOptimizer
        self.fm_separate_encoders = True
        self.fm_num_steps = 1
        self.fm_merger = tf.mul
        self.fm_activation = tf.sigmoid
        self.fm_lstm = False
        self.fm_train_set_size = 17000
        self.fm_num_iterations = 20000
        self.fm_expert_er_path = 'expert_data/er-2017-01-03-14-50-240T-sorted.bin'
        self.use_forward_model = True