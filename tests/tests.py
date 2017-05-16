
import hobo
import hobo.electrochemistry
import hobo.electrochemistry.data
import hobo.electrochemistry.models

import pylab as plt
import numpy as np
from math import sqrt

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
        'Cdl': 1.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53,
        'Ru': 8.0,
        'Cdl': 1.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.models.ECModel(dim_params)

    I,t = model.simulate(data.time)

    print 'distance = ',sqrt(np.sum(np.power(np.array(I)*model.I0-data.current,2))/np.sum(np.power(data.current,2)))

    plt.figure()
    plt.plot(data.time,data.current)
    plt.plot(t,I)
    plt.show()





