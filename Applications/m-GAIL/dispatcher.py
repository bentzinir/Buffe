import time
from driver import DRIVER


def dispatcher(environment, configuration, trained_model, sn_dir, expert_data):

    train_mode = not trained_model

    print '... Building controller'
    t0 = time.clock()

    driver = DRIVER(environment, configuration, trained_model, sn_dir, expert_data)

    itr = 0

    if train_mode:
        print 'Built controller in %0.2f [min]\n ... Training controller' % ((time.clock()-t0)/60)
    else:
        print 'Built controller in %0.2f [min]\n ... Playing saved model %s' % ((time.clock()-t0)/60, trained_model)

    while itr < configuration.n_train_iters:

        # test
        if itr % configuration.test_interval == 0:
            # display a test trajectory
            driver.test_step(itr)

            # print info line
            driver.print_info_line(itr)

            # save snapshot
            if train_mode:
                driver.save_model(itr)

        # train
        if train_mode:
            driver.train_step(itr)

        itr += 1
