import common

env_name = 'humanoid'
git_path = '/home/llt_lab/Documents/repo/'
run_dir = git_path + '/Buffe-2017/Applications/mgail/environments/' + env_name + '/'
env = __import__('environment').ENVIRONMENT(run_dir)


forward_model = __import__('forward_model').ForwardModel(state_size=env.state_size,
                                                              action_size=env.action_size,
                                                              rho=env.fm_rho,
                                                              beta=env.fm_beta,
                                                              encoding_size=env.fm_encoding_size,
                                                              batch_size=env.fm_batch_size,
                                                              multi_layered_encoder=env.fm_multi_layered_encoder,
                                                              num_steps=env.fm_num_steps,
                                                              separate_encoders=env.fm_separate_encoders,
                                                              merger=env.fm_merger,
                                                              activation=env.fm_activation,
                                                              lstm=env.fm_lstm)


forward_model.pretrain(env.fm_opt, env.fm_lr, env.fm_batch_size, env.fm_num_iterations, env.run_dir + env.fm_expert_er_path)
