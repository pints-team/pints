import numpy as np

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
        time = exp_data[:,0]
        current = exp_data[:,1]
    else:
        time = exp_data[:,2]
        current= exp_data[:,1]
    return time,current

