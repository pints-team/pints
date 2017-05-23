
import hobo
import hobo.electrochemistry
import hobo.electrochemistry.data
import hobo.electrochemistry.models

import pylab as plt
import numpy as np
from math import sqrt
import pystan

def test_ec_model():
    time,current = hobo.electrochemistry.data.read_cvsin_type_1('../../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt')
    assert time[0] == 0
    assert current[0] == 2.464073e-05

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

    #non dimensionalise data
    time = time/model.T0
    current = -current/model.I0

    # calculate model at time points given by the data file
    I,t = model.simulate(use_times=time)

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

    assert 0.1 > sqrt(np.sum(np.power(I-current,2))/np.sum(np.power(current,2)))



