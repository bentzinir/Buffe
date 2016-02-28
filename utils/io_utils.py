import time
from pprint import pprint
import os

def init_files(game_params, arch_params, solver_params):

    timestr = time.strftime("%Y-%m-%d")

    buf = ('%s_w%d_D%d_dt%.2f_delay0') % (game_params['force'][1], game_params['width'], game_params['D'], game_params['dt'])

    run_dir = game_params['run_path']+timestr

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    fid_log = init_log_file(run_dir, game_params, arch_params, solver_params)

    return run_dir, fid_log

def init_log_file(run_dir, game_params, arch_params, solver_params):

    f = open(run_dir + '/buffe.log','w', 0)
    f.write('######### Data params #########\n')
    pprint(game_params, stream=f)
    f.write('\n\n######### Arch params #########\n')
    pprint(arch_params, stream=f)
    f.write('\n\n######### Solver params #########\n')
    pprint(solver_params, stream=f)
    f.write('\n\n')

    return f
