import cma
import numpy as np
import numpy.random as rand
from copy import copy
from math import log,exp
import numdifftools as nd

class Prior:
    def __init__(self):
        self.data = {}
        self.n = 0
        self.mean = np.array([],float)
        self.variance = np.array([],float)

    def add_uniform_parameter(self, name, min_value, max_value):
        print 'Prior: adding uniform parameter ',name,' from (',min_value,'--',max_value,')'
        self.data[name] = ('uniform',min_value,max_value)
        self.n = len(self.data)
        np.append(self.mean,[0.5])
        np.append(self.variance,[1.0/12.0])

    def get_parameter_names(self):
        ret = []
        for i in self.data:
            ret.append(i)
        return ret

    def scale_sample_periodic(self, vector):
        assert(len(vector)==len(self.data))
        scaled = abs(vector) % 2
        for i,j in zip(self.data,range(scaled.size)):
            dist_name,min_val,max_val = self.data[i]
            if dist_name == 'uniform':
                if (scaled[j] <= 1):
                    scaled[j] = min_val + (max_val-min_val)*scaled[j]
                else:
                    scaled[j] = max_val - (max_val-min_val)*(scaled[j]-1)

        return scaled

    def scale_sample(self, vector):
        scaled = np.empty(len(vector),float)
        assert(len(vector)==len(self.data))
        for i,j in zip(self.data,range(len(scaled))):
            dist_name,min_val,max_val = self.data[i]
            if dist_name == 'uniform':
                scaled[j] = min_val + (max_val-min_val)*vector[j]

        return scaled

    def inv_scale_sample(self, vector):
        unscaled = np.empty(len(vector),float)
        assert(len(vector)==len(self.data))
        for i,j in zip(self.data,range(len(unscaled))):
            dist_name,min_val,max_val = self.data[i]
            if dist_name == 'uniform':
                unscaled[j] = (vector[j]-min_val)/(max_val-min_val)

        return unscaled


def fit_model_with_cmaes(data, model, prior):
    print 'fitting model with cmaes'
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
    return fitted_params

def mcmc_with_adaptive_covariance(data, model, prior, guess=None, burn_in=0, n_samples_mult=10000):

    n_samples = n_samples_mult*(prior.n+1)

    names = prior.get_parameter_names()

    sample_covariance = np.zeros((prior.n+1,prior.n+1),float)
    np.fill_diagonal(sample_covariance,np.zeros(prior.n+1)+(1.0/12.0))
    theta = np.zeros(prior.n+1)+0.5

    if guess is not None:
        assert len(guess) == prior.n
        print 'using guess: ',guess
        def error_func(v):
            sample_params = prior.scale_sample(v)
            model.set_params_from_vector(sample_params,names)
            current,time = model.simulate(use_times=data.time)
            return data.distance(current)

        theta[:-1] = prior.inv_scale_sample(guess)
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

    # adaptive covariance
    mu = best_theta
    log_a = 0
    theta = np.empty((prior.n+1,n_samples),float)
    theta[:,0] = best_theta
    total_accepted = 0
    for s in range(1,n_samples-1):
        if (s % (n_samples/10) == 0) and s != 0:
            print s/(n_samples/100),'% complete, accepted ',100*total_accepted/s,'%'
        gamma_s = (s+1.0)**-0.6
        #print sample_covariance
        theta_s = rand.multivariate_normal(theta[:,s],exp(log_a)*sample_covariance)
        if np.all(theta_s >= 0) and np.all(theta_s <= 1):
            log_likelihood_theta_s = log_likelihood(theta_s)
            log_alpha = log_likelihood_theta_s - log_likelihood_theta
            if log_alpha > 0 or rand.uniform() < exp(log_alpha):
                accepted = 1
                theta[:,s+1] = theta_s
            else:
                accepted = 0
                theta[:,s+1] = theta[:,s]
        else:
            accepted = 0
            theta[:,s+1] = theta[:,s]

        tmp = theta[:,s+1] - mu
        sample_covariance = (1-gamma_s)*sample_covariance + gamma_s*np.outer(tmp,tmp)
        mu = (1-gamma_s)*mu + gamma_s*theta[:,s+1]
        log_a += gamma_s*(accepted-0.25)
        total_accepted += accepted

    print 'accepted ',total_accepted,' out of ',n_samples,' samples'

    return theta









