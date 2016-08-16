import time
import common
from driver import DRIVER


def dispatcher(env):

    train_mode = not env.trained_model

    print '... Building controller'
    t0 = time.clock()

    driver = DRIVER(env)

    itr = 0

    if env.expert_data is None:
        driver.env.record_expert(driver.algorithm.er_expert)
        common.save_er(directory=env.run_dir,
                       module=driver.algorithm.er_expert)
        return

    if train_mode:
        print 'Built controller in %0.2f [min]\n ... Training controller' % ((time.clock()-t0)/60)
    else:
        print 'Built controller in %0.2f [min]\n ... Playing saved model %s' % ((time.clock()-t0)/60, env.trained_model)

    while itr < env.n_train_iters:

        # test
        if itr % env.test_interval == 0:

            # test
            driver.test_step()

            # display a test trajectory
            # driver.env.play_trajectory()
            driver.env.play_trajectory(expert=driver.algorithm.er_expert)

            # print info line
            driver.print_info_line(itr)

            # save snapshot
            if train_mode:
                driver.save_model(itr)

        # train
        if train_mode:
            driver.train_step(itr)

        itr += 1
