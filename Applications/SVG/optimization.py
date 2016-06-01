import time
from ctrlr_optimizer import CTRL_OPTMZR

def optimization(game_params, arch_params, solver_params, trained_model, sn_dir):

    train_mode = not trained_model

########### CONTROLLER 1 ###########
    print '... Building controller'
    t0 = time.clock()

    ctrl_optimizer = CTRL_OPTMZR(game_params, arch_params,solver_params, trained_model, sn_dir)

    iter = 0

    if train_mode:
        print 'Built controller in %0.2f [min]\n ... Training controller' % ((time.clock()-t0)/60)
    else:
        print 'Built controller in %0.2f [min]\n ... Playing saved model %s' % ((time.clock()-t0)/60 , trained_model)

    while (iter < solver_params['n_train_iters']):

        # test
        if iter % solver_params['test_interval'] == 0:
            # display a test tranjectory
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