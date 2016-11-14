import h5py
import sys
sys.path.append('/home/nir/work/git/Buffe/Applications')
# from mgail.ER import ER
from ER import ER
import common
import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=160)

if __name__ == '__main__':

    mgail_env_path = '/home/nir/work/git/Buffe/Applications/mgail/environments/'
    gail_data_path = '/home/nir/work/git/Buffe/Applications/imitation/imitation_runs/'
    # name = '/classic/trajs/trajs_cartpole.h5'; env = 'cartpole/'
    # name = '/classic/trajs/trajs_mountaincar.h5'; env = 'mountaincar/'
    # name = '/humanoid/trajs/trajs_humanoid.h5'
    # name = '/modern_stochastic/trajs/trajs_ant.h5'; env = 'ant/'
    # name = '/modern_stochastic/trajs/trajs_hopper.h5'; env = 'hopper/'
    name = '/modern_stochastic/trajs/trajs_halfcheetah.h5'; env = 'halfcheetah/'
    # name = '/modern_stochastic/trajs/trajs_walker.h5'; env = 'walker/'

    limit_trajs = None
    N_trajs = 50

    # load trajs
    with h5py.File(gail_data_path + name, 'r') as f:
        print('List of arrays in this file: \n', f.keys())
        full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
        dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
        exr_B_T = f['r_B_T'][:dset_size,...][...]
        exlen_B = f['len_B'][:dset_size,...][...]

    # save to ER
    state_size = exobs_B_T_Do.shape[2]
    action_size = exa_B_T_Da.shape[2]

    er = ER(memory_size=300000,
            state_dim=state_size,
            action_dim=action_size,
            reward_dim=1,  # stub connection
            batch_size=32,
            history_length=1)

    all_states = np.zeros((1, state_size))
    all_actions = np.zeros((1, action_size))
    sorted_trajs = (-exr_B_T).sum(axis=1).argsort()

    min_diff = 1000

    for i in range(N_trajs):
        traj_obs = exobs_B_T_Do[sorted_trajs[i]]
        traj_actions = exa_B_T_Da[sorted_trajs[i]]
        traj_rewards = exr_B_T[sorted_trajs[i]]
        traj_time = exlen_B[sorted_trajs[i]]

        next_states = traj_obs[1:traj_time-2]
        actions = traj_actions[:traj_time-3]
        er.add(actions=actions,
               rewards=np.ones(traj_time-1),
               next_states=next_states,
               terminals=np.zeros(traj_time-1))

        diff = (np.sqrt((np.diff(next_states, axis=0)) ** 2)).sum(axis=1).min()
        if diff < min_diff:
            min_diff = diff

        last_state = traj_obs[traj_time-1]
        last_action = traj_actions[traj_time-2]
        er.add(actions=[last_action], rewards=[1.], next_states=[last_state], terminals=[1])

        all_states = np.append(all_states, next_states, axis=0)
        all_actions = np.append(all_actions, actions, axis=0)

    er.states_mean = np.mean(all_states, axis=0)
    er.actions_mean = np.mean(all_actions, axis=0)

    er.states_std = np.std(all_states, axis=0)
    er.actions_std = np.std(all_actions, axis=0)

    common.save_er(directory=mgail_env_path + env, module=er)
