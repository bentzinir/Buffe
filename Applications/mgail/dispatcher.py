import time
import numpy as np
from driver import DRIVER


def dispatcher(env):

    print '... Building controller'
    t0 = time.clock()

    if env.expert_data is None:
        env.record_expert()
        return

    driver = DRIVER(env)

    if env.train_mode:
        print 'Built controller in %0.2f [min]\n ... Training controller' % ((time.clock()-t0)/60)
    else:
        print 'Built controller in %0.2f [min]\n ... Playing saved model %s' % ((time.clock()-t0)/60, env.trained_model)

    while driver.itr < env.n_train_iters:

        # test
        if driver.itr % env.test_interval == 0:

            # measure performance
            good_model = False
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=1, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            driver.reward = sum(R)/len(R)
            driver.reward_std = np.std(R)

            if driver.reward > env.success_th:
                good_model = True

            if driver.reward > driver.best_reward:
                driver.best_reward = driver.reward

            if driver.itr > env.kill_itr and driver.best_reward < env.reward_kill_th:
                break

            # print info line
            driver.print_info_line('full')

            # save snapshot
            if env.train_mode and (env.save_models or good_model):
                driver.save_model(dir_name=env.config_dir)

        # train
        if env.train_mode:
            driver.train_step()

        driver.itr += 1
