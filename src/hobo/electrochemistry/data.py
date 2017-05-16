import numpy as np
from hobo_cpp import hobo_vector

def read_cvsin_type_1(filename):
    """ read in a datafile of format svsin_type_1

    Args:
        filename (str): filename of the data file

    returns:
        time (numpy vector): vector of time samples
        current (numpy vector): vector of current samples
    """

    print 'Data: loading data from filename = ',filename,' ...'
    exp_data = np.loadtxt(filename,skiprows=19)
    print 'Data: done loading data.'
    if filename[-11:] == '_cv_current':
        t_index = 0
        c_index = 1
        time = exp_data[:,0]
        current = exp_data[:,1]
    else:
        t_index = 2
        c_index = 1
        time = exp_data[:,2]
        current= exp_data[:,1]
    n = exp_data.shape[0]
    print 'size = ',n
    time = hobo_vector()
    time[0:n] = exp_data[:,t_index]
    current = hobo_vector()
    current[0:n] = exp_data[:,c_index]
    return time,current

