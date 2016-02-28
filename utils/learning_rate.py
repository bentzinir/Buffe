import theano as t
import theano.tensor as tt

def create_learning_rate_func(solver_params):
    base = tt.fscalar('base')
    gamma = tt.fscalar('gamma')
    power = tt.fscalar('power')
    itrvl = tt.fscalar('itrvl')
    iter = tt.scalar('iter')

    if solver_params['lr_type']=='inv':
        lr_ = base * tt.pow(1 + gamma * iter, -power)

        lr = t.function(
            inputs=[iter, t.Param(base, default=solver_params['base']), t.Param(gamma, default=solver_params['gamma']), t.Param(power, default=solver_params['power'])],
            outputs=lr_)

    elif solver_params['lr_type']=='fixed':
        lr_ = base

        lr = t.function(
            inputs=[iter, t.Param(base, default=solver_params['base'])],
            outputs=lr_,
            on_unused_input='ignore')

    elif solver_params['lr_type']=='episodic':
        lr_ = base / (tt.floor(iter/itrvl) + 1)

        lr = t.function(
            inputs=[iter, t.Param(base, default=solver_params['base']), t.Param(itrvl, default=solver_params['interval'])],
            outputs=lr_,
            on_unused_input='ignore')
    return lr

