import cma
import numpy as np
import math
import multiprocessing
import itertools

def sample(v,data,model,prior,min_theta,max_theta,names):
    scaled_theta = abs(v[-1]) % 2
    if (scaled_theta <= 1):
        scaled_theta = min_theta + (max_theta-min_theta)*scaled_theta
    else:
        scaled_theta = max_theta - (max_theta-min_theta)*(scaled_theta-1)
    sample_params = prior.scale_sample_periodic(v[:-1],names)
    model.set_params_from_vector(sample_params,names)
    current,time = model.simulate(use_times=data.time)
    noise_log_likelihood = -math.log(scaled_theta)
    data_log_likelihood = data.log_likelihood(current,scaled_theta**2)/len(current)
    param_log_likelihood = prior.log_likelihood(sample_params,names)
    return -(noise_log_likelihood + data_log_likelihood + param_log_likelihood)


def sample_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return sample(*a_b)


def fit_model_with_cmaes(data, model, prior, IPOP=None):
    print '------------------------'
    print 'fitting model with cmaes'
    print '------------------------'
    x0 = np.zeros(prior.n+1)+0.5
    names = prior.get_parameter_names()

    max_current = np.max(data.current)
    min_theta = 0.005*max_current;
    max_theta = 0.03*max_current;


    if IPOP is None:
        es = cma.CMAEvolutionStrategy(x0, 0.25)
        print 'starting pool with',es.popsize,'workers'
        pool = multiprocessing.Pool(min(es.popsize,multiprocessing.cpu_count()-1))
        while not es.stop():
            X = es.ask()
            f_values = pool.map(sample_star,itertools.izip(X,
                                                 itertools.repeat(data),
                                                 itertools.repeat(model),
                                                 itertools.repeat(prior),
                                                 itertools.repeat(min_theta),
                                                 itertools.repeat(max_theta),
                                                 itertools.repeat(names)
                                                 ))
            # use chunksize parameter as es.popsize/len(pool)?
            es.tell(X, f_values)
            es.disp()
            #es.logger.add()
    else:
        bestever = cma.BestSolution()
        for lam in IPOP:
            print 'starting pool with',lam,'workers'
            pool = multiprocessing.Pool(min(lam,multiprocessing.cpu_count()-1))
            es = cma.CMAEvolutionStrategy(x0,  # 9-D
                                           0.25,  # initial std sigma0
                                           {'popsize': lam,  # options
                                            'verb_append': bestever.evalsall})
            while not es.stop():
                X = es.ask()    # get list of new solutions
                f_values = pool.map(sample_star,itertools.izip(X,
                                                 itertools.repeat(data),
                                                 itertools.repeat(model),
                                                 itertools.repeat(prior),
                                                 itertools.repeat(min_theta),
                                                 itertools.repeat(max_theta),
                                                 itertools.repeat(names)
                                                 ))

                es.tell(X, f_values) # besides for termination only the ranking in fit is used

                # display some output
                es.disp()  # uses option verb_disp with default 100

            print('termination:', es.stop())
            cma.pprint(es.best.__dict__)

            bestever.update(es.best)

    res = es.result()
    v = res[0]
    fitted_params = np.append(prior.scale_sample_periodic(v[:-1],names),[v[-1]*(max_theta-min_theta)+min_theta])
    objective_function = res[1]
    nevals = res[3]
    model.set_params_from_vector(fitted_params[:-1],names)
    print 'cmaes finished with objective function = ',objective_function,' and nevals = ',nevals, ' and noise estimate of ',fitted_params[-1]
    print '------------------------'
    return fitted_params

