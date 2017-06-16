import numpy as np
import numpy.random as rand
import math
import multiprocessing
import time
import os


def hierarchical_gibbs_sampler(names,datas,model,prior):
    assert len(samples) == len(datas)
    nexp = len(datas)
    ntheta = len(names)
    means_sample = np.zeros(ntheta)
    covariance_sample = np.zeros(ntheta)
    theta_sample = np.zeros(nexp,ntheta)
    names_index = np.zeros(ntheta,int)

    k_0 = 1
    nu_0 = 1
    mu_0 = model.get_params_from_vector(names)
    Gamma_0 = 1

    # find where names are in prior
    for i,name in zip(range(n),names):
        names_index[i] = prior.data.index(name)

    for i in range(n_samples):
        for j,data in zip(range(nexp),datas):
            # set prior for each name
            for ni in range(ntheta):
                prior.data.items[names_index[ni]] = Normal(means_sample[ni],covariance_sample[ni,ni])
            # inner mcmc
            samples = mcmc_with_adaptive_covariance(data,model,prior)
            # take a sample from this distribution
            theta_sample[j,:] = samples[rand.uniform(0,len(samples)),:]

        # sample from means distribution with a given stddev
        mu_hat = np.sum(theta_sample,axis=0)/nexp
        C = np.zeros(ntheta,ntheta)
        for j in range(nexp):
            tmp = theta_sample[j,:] - mu_hat
            C += np.outer(tmp,tmp)

        k = k_0 + nexp
        nu = nu_0 + nexp
        mu = (k_0*mu_0 + nexp*mu_hat)/k
        Gamma = Gamma_0 + C + (k_0*nexp)/k*np.outer(mu_hat-mu_0)

        # generate means_sample and covariance_sample from normal-inverse-Wishart
        sample_covariance = scipy.stats.invwishart.rvs(df=nu,scale=Gamma)
        means_sample = rand.multivariate_normal(mu,sample_covariance/k)




def mcmc_with_adaptive_covariance_chain(converged_pipes, means_pipes, variances_pipes, data, model, prior, burn_in_mult, n_samples_mult,results_send=None):
    master = isinstance(converged_pipes,list)
    if master:
        assert results_send is None
        print 'have',len(converged_pipes),'children'
    else:
        assert results_send is not None

    np.random.seed([os.getpid(),int(time.time())])

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
        noise_log_likelihood = -len(current)*math.log(sample_params[-1])
        data_log_likelihood = data.log_likelihood(current,sample_params[-1])
        param_log_likelihood = prior.log_likelihood(sample_params[:-1])
        return noise_log_likelihood + data_log_likelihood + param_log_likelihood

    log_likelihood_theta = log_likelihood(theta)


    # no adaptive covariance
    best_theta = theta
    best_log_likelihood = log_likelihood_theta
    for t in range(burn_in):
        if master and (t % (burn_in/10) == 0) and t != 0:
            print t/(burn_in/100),'% of burn-in complete'
        theta_s = rand.multivariate_normal(theta,sample_covariance)
        if np.all(theta_s >= 0) and np.all(theta_s <= 1):
            log_likelihood_theta_s = log_likelihood(theta_s)
            alpha = math.exp(log_likelihood_theta_s - log_likelihood_theta)
            if rand.uniform() < alpha:
                theta = theta_s
                log_likelihood_theta = log_likelihood_theta_s
                if (log_likelihood_theta > best_log_likelihood):
                    best_theta = theta
                    best_log_likelihood = log_likelihood_theta

    if master:
        print '100 % of burn-in complete'
        print '-------------------------'

    # adaptive covariance
    mu = best_theta
    log_a = 0
    thinning = 1
    theta_store = np.empty((n_samples/thinning,prior.n+1),float)
    theta_store[0,:] = mu
    theta = best_theta
    total_accepted = 0
    for s in range(1,n_samples-1):
        if (s % (n_samples/10) == 0) and s != 0:
            i = int(s/thinning)
            m = 2
            n = int(i-3*i/4)
            means = np.stack((np.mean(theta_store[i/2:3*i/4,:],axis=0),
                              np.mean(theta_store[3*i/4:i  ,:],axis=0)),axis=0)
            variances = np.stack((np.var(theta_store[i/2:3*i/4,:],ddof=1,axis=0),
                                  np.var(theta_store[3*i/4:i  ,:],ddof=1,axis=0)),axis=0)
            if not master:
                means_pipes.send(means)
                variances_pipes.send(variances)
                if converged_pipes.recv():
                    results_send.send(theta_store[i/2:i,:])
                    return
            else:
                for means_pipe,variances_pipe in zip(means_pipes,variances_pipes):
                    means = np.concatenate([means,means_pipe.recv()],axis=0)
                    variances = np.concatenate([variances,variances_pipe.recv()],axis=0)
                B = n*np.var(means,ddof=1,axis=0)
                W = np.mean(variances,axis=0)
                var_hat = (n-1.0)/n * W + 1.0/n * B
                R_hat = np.sqrt(var_hat/W)
                print s/(n_samples/100),'% complete, accepted ',1000*total_accepted/n_samples,'% R_hat = ',R_hat
                if np.max(np.abs(R_hat - 1)) < 1e-1:
                    for converged_pipe in converged_pipes:
                        converged_pipe.send(True)
                    return theta_store[i/2:i,:]
                else:
                    for converged_pipe in converged_pipes:
                        converged_pipe.send(False)

            total_accepted = 0

        gamma_s = (s+1.0)**-0.6
        theta_s = rand.multivariate_normal(theta,math.exp(log_a)*sample_covariance)
        accepted = 0
        if np.all(theta_s >= 0) and np.all(theta_s <= 1):
            log_likelihood_theta_s = log_likelihood(theta_s)
            log_alpha = log_likelihood_theta_s - log_likelihood_theta
            if log_alpha > 0 or rand.uniform() < math.exp(log_alpha):
                accepted = 1
                theta = theta_s

        if (s % thinning == 0):
            i = int(s/thinning)
            theta_store[i,:-1] = prior.scale_sample(theta[:-1])
            theta_store[i,-1] = theta[-1]*(max_theta-min_theta)+min_theta

        tmp = theta - mu
        sample_covariance = (1-gamma_s)*sample_covariance + gamma_s*np.outer(tmp,tmp)
        mu = (1-gamma_s)*mu + gamma_s*theta
        log_a += gamma_s*(accepted-0.25)
        total_accepted += accepted

    n = len(theta_store)
    if master:
        print '------------------------'
        return theta_store[n/2:,:]
    else:
        results_send.send(theta_store[n/2:,:])
        return




def mcmc_with_adaptive_covariance(data, model, prior, nchains=2, burn_in_mult=100, n_samples_mult=1000):
    print '-----------------------------'
    print 'MCMC with adaptive covariance'
    print '-----------------------------'

    model_params_save = model.get_params_from_vector(prior.get_parameter_names())
    ps = []
    converged_send_pipes = []
    results_recv_pipes = []
    means_recv_pipes = []
    variances_recv_pipes = []
    for i in range(nchains-1):

        means_recv, means_send = multiprocessing.Pipe(duplex=False)
        variances_recv, variances_send = multiprocessing.Pipe(duplex=False)
        converged_recv, converged_send = multiprocessing.Pipe(duplex=False)
        results_recv, results_send = multiprocessing.Pipe(duplex=False)
        converged_send_pipes.append(converged_send)
        results_recv_pipes.append(results_recv)
        means_recv_pipes.append(means_recv)
        variances_recv_pipes.append(variances_recv)

        p = multiprocessing.Process(target=mcmc_with_adaptive_covariance_chain,
                                    args=(converged_recv,means_send,variances_send,
                                          data,model,prior,
                                          burn_in_mult,n_samples_mult,results_send))
        p.start()
        ps.append(p)

    theta_store = mcmc_with_adaptive_covariance_chain(
                            converged_send_pipes,means_recv_pipes,variances_recv_pipes,
                            data,model,prior,burn_in_mult,n_samples_mult)

    print 'finished, with ',theta_store.shape[0],'samples'
    for p,results_recv in zip(ps,results_recv_pipes):
        theta_store = np.concatenate((theta_store,results_recv.recv()),axis=0)
        p.join()
    print 'got child samples, now have ',theta_store.shape[0],'samples'

    model.set_params_from_vector(model_params_save,prior.get_parameter_names())

    return theta_store



