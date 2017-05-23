import cma
import numpy as np

class Prior:
    def __init__(self):
        self.data = {}
        self.n = 0

    def add_uniform_parameter(self, name, min_value, max_value):
        print 'Prior: adding uniform parameter ',name,' from (',min_value,'--',max_value,')'
        self.data[name] = ('uniform',min_value,max_value)
        self.n = self.n + 1

    def get_parameter_names(self):
        ret = []
        for i in self.data:
            ret.append(i)
        return ret

    def get_mean(self):
        return np.zeros(self.n) + 0.5

    def scale_sample(self, vector):
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


def fit_model_with_cmaes(data, model, prior):
    x0 = prior.get_mean()
    names = prior.get_parameter_names()
    def sample(v):
        sample_params = prior.scale_sample(v)
        model.set_params_from_vector(sample_params,names)
        current,time = model.simulate(use_times=data.time)
        return data.distance(current)

    res = cma.fmin(sample,x0,0.25)
    fitted_params = prior.scale_sample(res[0])
    objective_function = res[1]
    nevals = res[3]
    model.set_params_from_vector(fitted_params,names)
    print 'cmaes finished with objective function = ',objective_function,' and nevals = ',nevals
    return objective_function,nevals


