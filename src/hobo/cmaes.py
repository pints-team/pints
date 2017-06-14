import cma
import numpy as np

def fit_model_with_cmaes(data, model, prior):
    print '------------------------'
    print 'fitting model with cmaes'
    print '------------------------'
    x0 = np.zeros(prior.n)+0.5
    names = prior.get_parameter_names()
    def sample(v):
        sample_params = prior.scale_sample_periodic(v)
        model.set_params_from_vector(sample_params,names)
        current,time = model.simulate(use_times=data.time)
        objective_func = data.distance(current)
        return objective_func

    res = cma.fmin(sample,x0,0.25)

    fitted_params = prior.scale_sample_periodic(res[0])
    objective_function = res[1]
    nevals = res[3]
    model.set_params_from_vector(fitted_params,names)
    print 'cmaes finished with objective function = ',objective_function,' and nevals = ',nevals
    print '------------------------'
    return fitted_params

