import tensorflow as tf
import numpy as np
import gym

class ENVIRONMENT(object):

    def __init__(self, run_dir):

        self.name = 'Hopper-v1'

        self.gym = gym.make(self.name)

        self._connect()

        self._train_params()

        self.run_dir = run_dir

    def _step(self, a):
        a = np.squeeze(a)
        self.t += 1
        state_ = self.state
        self.state, self.reward, self.done, self.info, self.qpos, self.qvel = self.gym.step(a)

        return self.state.astype(np.float32), np.float32(self.reward), self.done, self.qpos.astype(np.float32), self.qvel.astype(np.float32)

    def step(self, a, mode):
        qvel, qpos = [], []
        if mode == 'tensorflow':
            state, reward, done, qval, qpos = tf.py_func(self._step, inp=[a], Tout=[tf.float32, tf.float32, tf.bool, tf.float32, tf.float32], name='env_step_func')
            # DEBUG: flatten state. not sure if correctly
            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            state, reward, done, qvel, qpos = self._step(a)
        info = 0.
        return state, reward, done, info, qvel, qpos

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
        self.qpos_size = self.gym.model.data.qpos.shape[0]
        self.qvel_size = self.gym.model.data.qvel.shape[0]

    def _train_params(self):
        self.trained_model = None#'/home/llt_lab/Documents/repo/Buffe-fix/Applications/mgail/environments/hopper/snapshots/2017-02-16-03-33-050000.sn' #None #'/home/nir/work/git/Buffe/Applications/mgail/environments/halfcheetah/snapshots/2016-11-26-20-13-578000.sn'
        self.train_mode = True
        self.train_flags = [0, 1, 1]  # [autoencoder, transition, discriminator/policy]
        self.expert_data = 'expert_data/er-2017-02-16-04-23.bin'
        self.disc_as_classifier = True
        self.pre_load_buffer = False
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.kill_itr = self.n_train_iters
        self.reward_kill_th = -1
        self.test_interval = 1000
        self.n_steps_test = 1000
        self.vis_flag = False
        self.save_models = True
        self.config_dir = None
        self.continuous_actions = True
        self.tbptt = False
        self.success_th = 4500
        self.weight_decay = 1e-7
        self.save_agent_er = False
        self.save_agent_at_itr = 50000
        self.good_reward = 5000
        self.al_loss = 'CE'
        self.dad_gan = False

        # Main parameters to play with:
        self.er_agent_size = 50000
        self.reset_itrvl = 10000
        self.n_reset_iters = 10000
        self.model_identification_time = 1000
        self.prep_time = 1000
        self.collect_experience_interval = 15
        self.n_steps_train = 10
        self.discr_policy_itrvl = 100
        self.K_T = 1
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 70
        self.policy_al_w = 1e-2
        self.policy_tr_w = 1e-4
        self.policy_accum_steps = 7
        self.total_trans_err_allowed = 1000# 1e-0
        self.trans_loss_th = 50
        self.max_trans_iters = 2
        self.temp = 1.
        self.cost_sensitive_weight = 0.8
        self.biased_noise = 0

        self.t_size = [500, 300, 200]
        self.d_size = [200, 100]
        self.p_size = [100, 50]

        self.t_lr = 1e-2 #0.0005
        self.d_lr = 0.001
        self.p_lr = 0.0001

        self.w_std = 0.15

        self.noise_intensity = 6.
        self.do_keep_prob = 0.75

        self.smooth_over_steps = 50
        self.fm_lr = 1e-2
        self.fm_rho = 0.1
        self.fm_beta = 0.01
        self.fm_encoding_size = 100
        self.fm_batch_size = 300
        self.fm_multi_layered_encoder = True
        self.fm_opt = tf.train.AdamOptimizer
        self.fm_separate_encoders = True
        self.fm_num_steps = 1
        self.fm_merger = tf.mul
        self.fm_activation = tf.sigmoid
        self.fm_lstm = False
        self.fm_train_set_size = 17000
        self.fm_num_iterations = 2000
        self.fm_expert_er_path = \
                  '/home/llt_lab/Documents/repo/Buffe/Applications/mgail/environments/halfcheetah/expert_data/expert-2016-11-21-08-29-18T-sorted.bin'
        self.use_forward_model = True
