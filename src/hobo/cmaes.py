import cma
import numpy as np
import math

def fit_model_with_cmaes(data, model, prior):
    print '------------------------'
    print 'fitting model with cmaes'
    print '------------------------'
    x0 = np.zeros(prior.n+1)+0.5
    names = prior.get_parameter_names()

    max_current = np.max(data.current)
    min_theta = 0.005*max_current;
    max_theta = 0.03*max_current;

    def sample(v):
        scaled_theta = abs(v[-1]) % 2
        if (scaled_theta <= 1):
            scaled_theta = min_theta + (max_theta-min_theta)*scaled_theta
        else:
            scaled_theta = max_theta - (max_theta-min_theta)*(scaled_theta-1)
        sample_params = prior.scale_sample_periodic(v[:-1])
        model.set_params_from_vector(sample_params,names)
        current,time = model.simulate(use_times=data.time)
        noise_log_likelihood = -len(current)*math.log(scaled_theta)
        data_log_likelihood = data.log_likelihood(current,scaled_theta**2)
        param_log_likelihood = prior.log_likelihood(sample_params)
        return -(noise_log_likelihood + data_log_likelihood + param_log_likelihood)

    res = cma.fmin(sample,x0,0.25)

    v = res[0]
    fitted_params = np.append(prior.scale_sample_periodic(v[:-1]),[v[-1]*(max_theta-min_theta)+min_theta])
    objective_function = res[1]
    nevals = res[3]
    model.set_params_from_vector(fitted_params[:-1],names)
    print 'cmaes finished with objective function = ',objective_function,' and nevals = ',nevals, ' and noise estimate of ',fitted_params[-1]
    print '------------------------'
    return fitted_params[-1]

