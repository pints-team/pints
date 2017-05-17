from math import sqrt,pi
from hobo_cpp import e_implicit_exponential_mesh,hobo_map,hobo_vector

class ECModel:
    """Represents one electron transfer model in solution
            A + e- <-> B

    Args:
        params (dict): dictionary of parameters, containing these keys
                 'reversed', 'Estart','Ereverse','omega','phase','dE','v','T','a','c_inf','D'
                  'Ru',
                 'Cdl',
                 'E0',
                 'k0',
                 'alpha'
    """
    def __init__(self,params):
        try:
            print 'creating ECModel with parameters:'
            print '\treversed: ',params['reversed']
            print '\tEstart: ',params['Estart']
            print '\tEreverse: ',params['Ereverse']
            print '\tomega: ',params['omega']
            print '\tphase: ',params['phase']
            print '\tamplitude: ',params['dE']
            print '\tdc scan rate: ',params['v']
            print '\ttemperature: ',params['T']
            print '\telectrode area: ',params['a']
            print '\tc_inf: ',params['c_inf']
            print '\tdiffusion constant: ',params['D']
            print '\tRu: ',params['Ru']
            print '\tCdl: ',params['Cdl']
            print '\tE0: ',params['E0']
            print '\tk0: ',params['k0']
            print '\talpha: ',params['alpha']
        except NameError as e:
            print 'NameError: ',e.value

        self.dim_params = params

        if (params['reversed']):
            self.dim_params['E0'] = self.dim_params['Estart'] - (self.dim_params['E0'] - self.dim_params['Ereverse'])
            self.dim_params['Estart'],self.dim_params['Ereverse'] = self.dim_params['Ereverse'],self.dim_params['Estart']
            self.dim_params['v'] = -self.dim_params['v']

        self._E0, self._T0, self._L0, self._I0 = self._calculate_characteristic_values()

        self.params = hobo_map()
        self.params['Estart'] = self.dim_params['Estart']/self._E0
        self.params['Ereverse'] = self.dim_params['Ereverse']/self._E0
        self.params['omega'] = 2*pi*self.dim_params['omega']*self._T0
        if self.dim_params['reversed']:
            self.params['phase'] = self.dim_params['phase'] + pi
        else:
            self.params['phase'] = self.dim_params['phase']
        self.params['dE'] = self.dim_params['dE']/self._E0

        self.params['k0'] = self.dim_params['k0']*self._L0/self.dim_params['D']
        self.params['alpha'] = self.dim_params['alpha']
        self.params['E0'] = self.dim_params['E0']/self._E0
        self.params['Ru'] = self.dim_params['Ru']*self._I0/self._E0
        self.params['Cdl'] = self.dim_params['Cdl']*self.dim_params['a']*self._E0/(self._I0*self._T0)

    def simulate(self, use_times=None, use_current=None):
        params = self.params

        current = hobo_vector()
        if use_current is None:
            current = hobo_vector()
        elif type(use_current) is hobo_vector:
            current = use_current
        else:
            current = hobo_vector()
            current[0:len(use_current)] = use_current_

        if use_times is None:
            times = hobo_vector()
        elif type(use_times) is hobo_vector:
            times = use_times
        else:
            times = hobo_vector()
            times[0:len(use_times)] = use_times

        e_implicit_exponential_mesh(params,current,times)

        return current,times



    @property
    def E0(self):
        """E0 (double): characteristic voltage"""
        return self._E0

    @property
    def T0(self):
        """T0 (double): characteristic temperature"""
        return self._T0

    @property
    def I0(self):
        """I0 (double): characteristic current"""
        return self._I0

    @property
    def L0(self):
        """L0 (double): characteristic length"""
        return self._L0

    def _calculate_characteristic_values(self):

        v = self.dim_params['v']
        T = self.dim_params['T']
        a = self.dim_params['a']

        #Faraday constant (C mol-1)
        F = 96485.3328959
        #gas constant (J K-1 mol-1)
        R = 8.314459848

        E_0 = R*T/F
        T_0 = abs(E_0/v)

        D = self.dim_params['D']
        L_0 = sqrt(D*T_0)
        c_inf = self.dim_params['c_inf']
        I_0 = D*F*a*c_inf/L_0

        return E_0,T_0,L_0,I_0


