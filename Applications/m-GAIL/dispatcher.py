import time
import common
from driver import DRIVER


def dispatcher(environment, trained_model, run_dir, expert_data):

    train_mode = not trained_model

    print '... Building controller'
    t0 = time.clock()

    driver = DRIVER(environment, trained_model, run_dir, expert_data)

    itr = 0

    if expert_data is None:
        driver.env.record_expert(driver.algorithm.er_expert)
        common.save_er(directory=run_dir+'experts/',
                       env_name=environment.name,
                       module=driver.algorithm.er_expert)
        return

    if train_mode:
        print 'Built controller in %0.2f [min]\n ... Training controller' % ((time.clock()-t0)/60)
    else:
        print 'Built controller in %0.2f [min]\n ... Playing saved model %s' % ((time.clock()-t0)/60, trained_model)

    while itr < environment.n_train_iters:

        # test
        if itr % environment.test_interval == 0:

            # test
            driver.test_step()

            # display a test trajectory
            driver.env.play_trajectory()

            # print info line
            driver.print_info_line(itr)

            # save snapshot
            if train_mode:
                driver.save_model(itr)

        # train
        if train_mode:
            driver.train_step(itr)

        itr += 1
