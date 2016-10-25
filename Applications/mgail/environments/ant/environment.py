import tensorflow as tf
import numpy as np
import gym

class ENVIRONMENT(object):

    def __init__(self, run_dir):

        self.name = 'Ant-v1'

        self.gym = gym.make(self.name)

        self._connect()

        self._train_params()

        self.run_dir = run_dir

    def _step(self, a):
        a = np.squeeze(a)
        self.t += 1
        self.state, self.reward, self.done, self.info = self.gym.step(a)
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
        self.trained_model = None
        self.train_mode = True
        self.train_flags = [0, 1, 1]  # [autoencoder, transition, discriminator/policy]
        self.expert_data = 'expert_data/expert-2016-10-20-09-49-25T-sorted.bin'
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.test_interval = 2000
        self.n_steps_test = 1000
        self.vis_flag = True
        self.model_identification_time = 0
        self.save_models = True
        self.config_dir = None
        self.tbptt = False
        self.success_th = 3500

        # Main parameters to play with:
        self.collect_experience_interval = 30
        self.n_steps_train = 25
        self.discr_policy_itrvl = 500
        self.K_T = 2
        self.K_D = 1
        self.K_P = 1
        self.gamma = 0.99
        self.batch_size = 70
        self.policy_al_loss_w = 1e-0
        self.er_agent_size = 100000
        self.total_trans_err_allowed = 1000# 1e-0
        self.trans_loss_th = 2e-2
        self.max_trans_iters = 2

        self.t_size = [400, 200]
        self.d_size = [100, 50]
        self.p_size = [75, 25]
        self.t_lr = 0.0005
        self.d_lr = 0.001
        self.p_lr = 0.0005

        self.noise_intensity = 3.
        self.dropout_ratio = 0.4

        # Parameters i don't want to play with
        self.disc_as_classifier = True
        self.sae_hidden_size = 80
        self.sae_beta = 3.
        self.sae_rho = 0.01
        self.sae_batch = 25e3
        self.use_sae = False