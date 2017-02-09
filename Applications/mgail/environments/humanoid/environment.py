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
        self.state, self.reward, self.done, self.info, self.qpos, self.qvel = self.gym.step(a)

        return self.state.astype(np.float32), self.reward.astype(np.float32), self.done, self.qpos.astype(np.float32), self.qvel.astype(np.float32)

    def step(self, a, mode):
        if mode == 'tensorflow':
            state, reward, done, qpos, qvel = tf.py_func(self._step, inp=[a], Tout=[tf.float32, tf.float32, tf.bool, tf.float32, tf.float32], name='env_step_func')
            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            state, reward, done, qpos, qvel = self._step(a)
        info = 0.
        return state, reward, done, info, qpos, qvel

    def reset(self, qpos=None, qvel=None):
        self.t = 0
        self.state = self.gym.reset(qpos, qvel)
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
        self.qpos_size = self.gym.env.model.data.qpos.shape[0]
        self.qvel_size = self.gym.env.model.data.qvel.shape[0]

    def _train_params(self):
        self.trained_model = None
        self.train_mode = True
        self.train_flags = [0, 1, 1]  # [autoencoder, transition, discriminator/policy]
        self.expert_data = 'expert_data/er-2017-02-06-15-06-490k-fake.bin'
        self.pre_load_buffer = False
        self.n_train_iters = 2000000
        self.n_episodes_test = 5
        self.test_interval = 10000
        self.n_steps_test = 1000
        self.good_reward = 8000
        self.vis_flag = False
        self.save_models = False
        self.save_agent_er = False
        self.save_agent_at_itr = 50000
        self.config_dir = None
        self.continuous_actions = True
        self.weight_decay = 1e-6
        self.al_loss = 'CE'
        self.dad_gan = False

        # Main parameters to play with:
        self.er_agent_size = 200000
        self.model_identification_time = 1000
        self.prep_time = 1000
        self.collect_experience_interval = 15
        self.n_steps_train = 10
        self.discr_policy_itrvl = 100
        self.K_T = 0
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 70
        self.policy_al_w = 1e-3
        # self.policy_tr_w = 5e-3
        self.policy_accum_steps = 5
        self.total_trans_err_allowed = 1.
        self.max_trans_iters = 2
        self.temp = 1.
        self.cost_sensitive_weight = 1.1

        self.t_size = [500, 300, 200]
        self.d_size = [1000, 500, 150]
        self.p_size = [600, 300]

        self.t_lr = 1e-2
        self.d_lr = 0.001
        self.p_lr = 0.0001

        self.w_std = 0.15

        self.noise_intensity = 6.
        self.do_keep_prob = 0.75

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
