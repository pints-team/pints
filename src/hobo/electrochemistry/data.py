import numpy as np
from hobo_cpp import hobo_vector
from math import pi,floor,ceil

def read_cvsin_type_1(filename):
    """ read in a datafile of format svsin_type_1

    Args:
        filename (str): filename of the data file

    returns:
        time (numpy vector): vector of time samples
        current (numpy vector): vector of current samples
    """

    exp_data = np.loadtxt(filename,skiprows=19)
    if filename[-11:] == '_cv_current':
        t_index = 0
        c_index = 1
    else:
        t_index = 2
        c_index = 1
    return exp_data[:,t_index],exp_data[:,c_index]


class ECTimeData:
    def __init__(self, filename, model, datafile_type='scsin_type_1'):
        print 'ECTimeData: loading data from filename = ',filename,' ...'
        self.time, self.current = read_cvsin_type_1(filename)
        print '\tdone loading data.'

        dt = self.time[100]-self.time[99]
        samples_per_period = 1.0/(model.dim_params['omega']*dt)
        downsample = int(floor(samples_per_period/200.0))
        if downsample == 0:
            downsample = 1
        print '\tBefore downsampling, have ',len(self.time),' data points'
        print '\tDatafile has ',samples_per_period,' samples per period.'
        print '\tReducing number of samples using a moving average window of size ',downsample

        new_length = int(len(self.time)/downsample)
        pad_size = int(ceil(float(len(self.time))/downsample)*downsample) - len(self.time)
        self.time = np.append(self.time,np.zeros(pad_size)*np.NaN)
        self.time = np.nanmean(self.time.reshape(-1,downsample),axis=1)
        self.current = np.append(self.current,np.zeros(pad_size)*np.NaN)
        self.current = np.nanmean(self.current.reshape(-1,downsample),axis=1)

        self.current = self.current / model.I0
        self.time = self.time / model.T0
        self.distance_scale = np.linalg.norm(self.current)

        print '\tAfter downsampling, have ',len(self.time),' data points'

    def distance(self, current):
        return np.linalg.norm(current-self.current)/self.distance_scale
