import datetime
import time
import numpy as np
from ctrlr_optimizer import CTRL_OPTMZR

def optimization(game_params, arch_params, solver_params, trained_model, sn_dir):

    train_mode = not trained_model

########### CONTROLLER 1 ###########
    print '%s Building controller...' % datetime.datetime.now()
    t0 = time.clock()

    ctrl_optimizer = CTRL_OPTMZR(game_params, arch_params,solver_params, trained_model, sn_dir)

    ctrl_optimizer.init_theano_funcs()

    iter = 0

    if train_mode:
        print '%s Built controller in %0.2f [min]\n ... Training controller' % (datetime.datetime.now(), ((time.clock()-t0)/60))
    else:
        print '%s Built controller in %0.2f [min]\n ... Playing saved model %s' % (datetime.datetime.now(), (time.clock()-t0)/60 , trained_model)

    while (iter < solver_params['n_train_iters']):

        # test
        if iter % solver_params['test_interval'] == 0:

            # display a test trajectory
            ctrl_optimizer.test_step()

            # print info line
            ctrl_optimizer.print_info_line(iter)

            # save snapshot
            if train_mode:
                ctrl_optimizer.save_model(iter)

            # display a roll-out
            ctrl_optimizer.play_trajectory()

        # train
        if train_mode:
            ctrl_optimizer.train_step(iter)

        iter = iter + 1
