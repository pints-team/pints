import numpy as np

class Uniform:

    def __str__(self):
        return 'Uniform'

    def log_likelihood(self,value):
        return 0


class Normal:
    def __init__(self,mean,variance):
        self.mean = mean
        self.variance = variance

    def __str__(self):
        return 'Normal with N(mean,variance) = N(%f,%f)'%(self.mean,self.variance)

    def log_likelihood(self,value):
        return -(value-self.mean)**2/self.variance

class Prior:
    def __init__(self):
        self.data = {}
        self.n = 0

    def add_parameter(self, name, dist, min_value, max_value, quiet=False):
        if not quiet:
            print 'Prior: adding parameter ',name,':'
            print '\tprior distribution = ',dist
            print '\tbounds = (',min_value,'--',max_value,')'
        self.data[name] = (dist,min_value,max_value)
        self.n = len(self.data)

    def log_likelihood(self,vector,names):
        assert(len(vector)==len(names))
        ll = 0
        for name,j in zip(names,range(len(vector))):
            prior,min_v,max_v = self.data[name]
            ll += prior.log_likelihood(vector[j]);
        return ll

    def get_parameter_names(self):
        ret = []
        for key in self.data:
            ret.append(key)
        return ret

    def scale_sample_periodic(self, vector, names):
        assert(len(vector)==len(names))
        scaled = abs(vector) % 2
        for name,j in zip(names,range(scaled.size)):
            prior,min_v,max_v = self.data[name]
            if (scaled[j] <= 1):
                scaled[j] = min_v + (max_v-min_v)*scaled[j]
            else:
                scaled[j] = max_v - (max_v-min_v)*(scaled[j]-1)
        return scaled

    def scale_sample(self, vector, names):
        assert(len(vector)==len(names))
        scaled = np.empty(len(vector),float)
        assert(len(vector)==len(self.data))
        for name,j in zip(names,range(scaled.size)):
            prior,min_v,max_v = self.data[name]
            scaled[j] = min_v + (max_v-min_v)*vector[j]
        return scaled

    def inv_scale_sample(self, vector, names):
        assert(len(vector)==len(names))
        unscaled = np.empty(len(vector),float)
        assert(len(vector)==len(self.data))
        for name,j in zip(names,range(unscaled.size)):
            prior,min_v,max_v = self.data[name]
            unscaled[j] = (vector[j]-min_v)/(max_v-min_v)
        return unscaled





