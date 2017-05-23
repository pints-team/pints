
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
    data = hobo.electrochemistry.data.ECTimeData(filename,model)

    # calculate model at time points given by the data file
    I,t = model.simulate(use_times=data.time)

    plt.figure()
    plt.plot(t,I)
    plt.plot(data.time,data.current)
    plt.show(block=True)

    error = sqrt(np.sum(np.power(data.current-I,2))/np.sum(np.power(data.current,2)))
    print 'error = ',error
    assert 0.11 > error

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

    data = hobo.electrochemistry.data.ECTimeData(filename,model)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_uniform_parameter('E0',model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_uniform_parameter('k0',0,10000)
    prior.add_uniform_parameter('Cdl',0,20)
    prior.add_uniform_parameter('omega',model.params['omega']*0.99,model.params['omega']*1.01)
    prior.add_uniform_parameter('phase',model.params['phase']-pi/10,model.params['phase']+pi/10)

    objective_func,nevals = hobo.fit_model_with_cmaes(data,model,prior)

    assert objective_func < 0.11

    # calculate model at time points given by the data file
    I,t = model.simulate(use_times=data.time)

    assert 0.1 > sqrt(np.sum(np.power(I-current,2))/np.sum(np.power(current,2)))



# calculate model using stan
    # sm = model.get_stan_model()

    # combined_params = dict(T=len(time),ts=time)
    # combined_params.update(model.nondim_params)
    # op = sm.sampling(data=combined_params, chains=1, iter=1, warmup=0,algorithm='Fixed_param')

    # stan_current = op.extract('Itot')

    # f = plt.figure()
    # plt.plot(time,current,label='exp')
    # plt.plot(t,I,label='cpp')
    # plt.show()
    # plt.plot(time,stan_current,label='stan')



if __name__ == "__main__":
    test_ec_model()
