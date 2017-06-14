import numpy as np
import numpy.random as rand

def mcmc_with_adaptive_covariance(data, model, prior, burn_in_mult=1000, n_samples_mult=10000):
    print '-----------------------------'
    print 'MCMC with adaptive covariance'
    print '-----------------------------'
    n_samples = n_samples_mult*(prior.n+1)
    burn_in = burn_in_mult*(prior.n+1)

    names = prior.get_parameter_names()

    sample_covariance = np.zeros((prior.n+1,prior.n+1),float)
    np.fill_diagonal(sample_covariance,np.zeros(prior.n+1)+(1.0/12.0))
    theta = np.zeros(prior.n+1)+0.5

    # get initial values from model
    for i,name in zip(range(prior.n),names):
        theta[i] = model.params[name]

    theta[:-1] = prior.inv_scale_sample(theta[:-1])
        #print 'using guess: ',guess
        #def error_func(v):
        #    sample_params = prior.scale_sample(v)
        #    model.set_params_from_vector(sample_params,names)
        #    current,time = model.simulate(use_times=data.time)
        #    return data.distance(current)

        #theta[:-1] = prior.inv_scale_sample(guess)
        #hessian = nd.Hessian(error_func)(theta[:-1])
        #sample_covariance[:-1,:-1] = np.linalg.inv(hessian)
        #print 'using sample covariance: ',sample_covariance


    # get maximum current from data, for noise prior
    max_current = np.max(data.current)
    min_theta = 0.005*max_current;
    max_theta = 0.03*max_current;

    def log_likelihood(v):
        sample_params = np.append(prior.scale_sample(v[:-1]),[v[-1]*(max_theta-min_theta)+min_theta])
        model.set_params_from_vector(sample_params[:-1],names)
        current,time = model.simulate(use_times=data.time)
        diff = current - data.current
        return -len(current)*log(sample_params[-1]) - (0.5/sample_params[-1]**2)*np.inner(diff,diff)

    log_likelihood_theta = log_likelihood(theta)


    # no adaptive covariance
    best_theta = theta
    best_log_likelihood = log_likelihood_theta
    for t in range(burn_in):
        if (t % (burn_in/10) == 0) and t != 0:
            print t/(burn_in/100),'% of burn-in complete'
        theta_s = rand.multivariate_normal(theta,sample_covariance)
        if np.all(theta_s >= 0) and np.all(theta_s <= 1):
            log_likelihood_theta_s = log_likelihood(theta_s)
            alpha = exp(log_likelihood_theta_s - log_likelihood_theta)
            if rand.uniform() < alpha:
                theta = theta_s
                log_likelihood_theta = log_likelihood_theta_s
                if (log_likelihood_theta > best_log_likelihood):
                    best_theta = theta
                    best_log_likelihood = log_likelihood_theta

    print '100 % of burn-in complete'

    # adaptive covariance
    mu = best_theta
    log_a = 0
    thinning = 10
    theta_store = np.empty((n_samples/thinning,prior.n+1),float)
    theta_store[0,:] = mu
    theta = best_theta
    total_accepted = 0
    for s in range(1,n_samples-1):
        if (s % (n_samples/10) == 0) and s != 0:
            print s/(n_samples/100),'% complete, accepted ',1000*total_accepted/n_samples,'%'
            total_accepted = 0
        gamma_s = (s+1.0)**-0.6
        #print sample_covariance
        theta_s = rand.multivariate_normal(theta,exp(log_a)*sample_covariance)
        accepted = 0
        if np.all(theta_s >= 0) and np.all(theta_s <= 1):
            log_likelihood_theta_s = log_likelihood(theta_s)
            log_alpha = log_likelihood_theta_s - log_likelihood_theta
            if log_alpha > 0 or rand.uniform() < exp(log_alpha):
                accepted = 1
                theta = theta_s

        if (s % thinning == 0):
            theta_store[s/thinning,:] = theta

        tmp = theta - mu
        sample_covariance = (1-gamma_s)*sample_covariance + gamma_s*np.outer(tmp,tmp)
        mu = (1-gamma_s)*mu + gamma_s*theta
        log_a += gamma_s*(accepted-0.25)
        total_accepted += accepted
    print '------------------------'

    return theta_store






