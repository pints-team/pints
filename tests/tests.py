import hobo
import hobo.electrochemistry
import pickle

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

    model = hobo.electrochemistry.ECModel(dim_params)
    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

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

    model = hobo.electrochemistry.ECModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',hobo.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    prior.add_parameter('Cdl',hobo.Uniform(),0,20)

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

    model = hobo.electrochemistry.ECModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',hobo.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    prior.add_parameter('Cdl',hobo.Uniform(),0,20)
    prior.add_parameter('omega',hobo.Uniform(),model.params['omega']*0.99,model.params['omega']*1.01)
    prior.add_parameter('phase',hobo.Uniform(),model.params['phase']-pi/10,model.params['phase']+pi/10)

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

    model = hobo.electrochemistry.ECModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5,ignore_end_samples=0)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',hobo.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    prior.add_parameter('Cdl',hobo.Uniform(),0,20)

    print 'before cmaes, parameters are:'
    names = prior.get_parameter_names()
    for name in prior.get_parameter_names():
        print name,': ',model.params[name]

    model.set_params_from_vector([0.00312014718956,2.04189332425,7.274953392],['Cdl','k0','E0'])
    #hobo.fit_model_with_cmaes(data,model,prior)


    pickle.dump( model , open( "model.p", "wb" ) )
    model = pickle.load( open( "model.p", "rb" ) )

    I,t = model.simulate(use_times=data.time)
    plt.figure()
    plt.plot(t,I)
    plt.plot(data.time,data.current)
    plt.savefig('test_mcmc_after_cmaes.pdf')
    plt.close()

    print 'after cmaes, parameters are:'
    names = prior.get_parameter_names()
    for name in prior.get_parameter_names():
        print name,': ',model.params[name]

    samples = hobo.mcmc_with_adaptive_covariance(data,model,prior)
    #samples = np.random.uniform(size=(1000,prior.n+1))

    fig,axes = hobo.scatter_grid(samples, model, prior)
    plt.savefig('test_mcmc_after_mcmc.pdf')
    #plt.show(block=True)




if __name__ == "__main__":
    test_mcmc()
