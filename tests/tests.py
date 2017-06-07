
import hobo
import hobo.electrochemistry
import hobo.electrochemistry.data
import hobo.electrochemistry.models

import numpy as np
from math import sqrt
import pystan
import cma
from math import pi
import pylab as plt

def test_ec_model():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

    dim_params = {
        'reversed': True,
        'Estart': 0.5,
        'Ereverse': -0.1,
        'omega': 9.0152,
        'phase': 0,
        'dE': 0.08,
        'v': -0.08941,
        't_0': 0.001,
        'T': 297.0,
        'a': 0.07,
        'c_inf': 1*1e-3*1e-3,
        'D': 7.2e-6,
        'Ru': 8.0,
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.models.ECModel(dim_params)
    data = hobo.electrochemistry.data.ECTimeData(filename,model,ignore_begin_samples=40)

    # calculate model at time points given by the data file
    I,t = model.simulate(use_times=data.time)

    plt.figure()
    plt.plot(t,I)
    plt.plot(data.time,data.current)
    plt.show(block=True)

    error = data.distance(I)
    print 'error = ',error
    assert 0.07 > error

def test_nonlin_op():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

    dim_params = {
        'reversed': True,
        'Estart': 0.5,
        'Ereverse': -0.1,
        'omega': 9.0152,
        'phase': 0,
        'dE': 0.08,
        'v': -0.08941,
        't_0': 0.001,
        'T': 297.0,
        'a': 0.07,
        'c_inf': 1*1e-3*1e-3,
        'D': 7.2e-6,
        'Ru': 8.0,
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.models.ECModel(dim_params)

    data = hobo.electrochemistry.data.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_uniform_parameter('E0',model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_uniform_parameter('k0',0,10000)
    prior.add_uniform_parameter('Cdl',0,20)
    #prior.add_uniform_parameter('omega',model.params['omega']*0.99,model.params['omega']*1.01)
    #prior.add_uniform_parameter('phase',model.params['phase']-pi/10,model.params['phase']+pi/10)

    params = hobo.fit_model_with_cmaes(data,model,prior)
    print params

    model.set_params_from_vector(params,prior.get_parameter_names())
    I,t = model.simulate(use_times=data.time)

    assert data.distance(I) < 0.07


def test_stan():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

    dim_params = {
        'reversed': True,
        'Estart': 0.5,
        'Ereverse': -0.1,
        'omega': 9.0152,
        'phase': 0,
        'dE': 0.08,
        'v': -0.08941,
        't_0': 0.001,
        'T': 297.0,
        'a': 0.07,
        'c_inf': 1*1e-3*1e-3,
        'D': 7.2e-6,
        'Ru': 8.0,
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.models.ECModel(dim_params)

    data = hobo.electrochemistry.data.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_uniform_parameter('E0',model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_uniform_parameter('k0',0,10000)
    prior.add_uniform_parameter('Cdl',0,20)
    prior.add_uniform_parameter('omega',model.params['omega']*0.99,model.params['omega']*1.01)
    prior.add_uniform_parameter('phase',model.params['phase']-pi/10,model.params['phase']+pi/10)

    # calculate model using stan
    sm = model.get_stan_model()

    combined_params = dict(T=len(data.time),ts=data.time)
    combined_params.update(model.nondim_params)
    op = sm.sampling(data=combined_params, chains=1, iter=1, warmup=0,algorithm='Fixed_param')
    print 'finished'

    stan_current = op.extract('Itot')

    f = plt.figure()
    plt.plot(data.time,data.current,label='exp')
    plt.plot(data.time,stan_current,label='stan')
    plt.show()

def test_mcmc():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

    dim_params = {
        'reversed': True,
        'Estart': 0.5,
        'Ereverse': -0.1,
        'omega': 9.0152,
        'phase': 0,
        'dE': 0.08,
        'v': -0.08941,
        't_0': 0.001,
        'T': 297.0,
        'a': 0.07,
        'c_inf': 1*1e-3*1e-3,
        'D': 7.2e-6,
        'Ru': 8.0,
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.models.ECModel(dim_params)

    data = hobo.electrochemistry.data.ECTimeData(filename,model,ignore_begin_samples=5,ignore_end_samples=0)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_uniform_parameter('E0',model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_uniform_parameter('k0',0,10000)
    prior.add_uniform_parameter('Cdl',0,20)
    #prior.add_uniform_parameter('omega',model.params['omega']*0.99,model.params['omega']*1.01)
    #prior.add_uniform_parameter('phase',model.params['phase']-pi/10,model.params['phase']+pi/10)

    #fitted_params,scaled_fitted_params,objective_func = hobo.fit_model_with_cmaes(data,model,prior)

    # choose initial guess as params we set above
    names = prior.get_parameter_names()
    init_guess = np.empty(prior.n,float)
    for i,name in zip(range(prior.n),names):
        init_guess[i] = model.params[name]

    samples = hobo.mcmc_with_adaptive_covariance(data,model,prior,guess=init_guess,n_samples_mult=1000)
    #samples = np.random.uniform(size=(prior.n,1000))

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    for ax,i in zip([ax1,ax2,ax3,ax4,ax5,ax6],range(prior.n+1)):
        print np.min(samples[i,:]),' ',np.max(samples[i,:])
        ax.hist(samples[i,:], 50, facecolor='green', alpha=0.75)
    plt.show(block=True)





if __name__ == "__main__":
    test_mcmc()
