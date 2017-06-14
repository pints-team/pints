import numpy as np

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





