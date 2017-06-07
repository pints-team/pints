from math import sqrt,pi
from hobo_cpp import e_implicit_exponential_mesh,hobo_map,hobo_vector
import pystan

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
            print 'creating ECModel with (dimensional) parameters:'
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

        self.E0, self.T0, self.L0, self.I0 = self._calculate_characteristic_values()

        self.params = hobo_map()
        self.params['Estart'] = self.dim_params['Estart']/self.E0
        self.params['Ereverse'] = self.dim_params['Ereverse']/self.E0
        self.params['omega'] = 2*pi*self.dim_params['omega']*self.T0
        if self.dim_params['reversed']:
            self.params['phase'] = self.dim_params['phase'] + pi
        else:
            self.params['phase'] = self.dim_params['phase']
        self.params['dE'] = self.dim_params['dE']/self.E0

        self.params['k0'] = self.dim_params['k0']*self.L0/self.dim_params['D']
        self.params['alpha'] = self.dim_params['alpha']
        self.params['E0'] = self.dim_params['E0']/self.E0
        self.params['Ru'] = self.dim_params['Ru']*abs(self.I0)/self.E0
        self.params['Cdl'] = self.dim_params['Cdl']*self.dim_params['a']*self.E0/(abs(self.I0)*self.T0)

        self._nondim_params = {}
        self._nondim_params['Estart'] = self.params['Estart']
        self._nondim_params['Ereverse'] = self.params['Ereverse']
        self._nondim_params['omega'] = self.params['omega']
        self._nondim_params['phase'] = self.params['phase']
        self._nondim_params['dE'] = self.params['dE']
        self._nondim_params['k0'] = self.params['k0']
        self._nondim_params['alpha'] = self.params['alpha']
        self._nondim_params['E0'] = self.params['E0']
        self._nondim_params['Ru'] = self.params['Ru']
        self._nondim_params['Cdl'] = self.params['Cdl']

    def dimensionalise(self,value,name):
        if name == 'Estart':
            return value*self.E0
        elif name == 'Ereverse':
            return value*self.E0
        elif name == 'omega':
            return value/(2*pi*self.T0)
        elif name == 'phase':
            return value
        elif name == 'dE':
            return value*self.E0
        elif name == 'k0':
            return value*self.dim_params['D']/self.L0
        elif name == 'alpha':
            return value
        elif name == 'E0':
            return value*self.E0
        elif name == 'Ru':
            return value*self.E0/self.I0
        elif name == 'Cdl':
            return value*self.I0*self.T0/(self.dim_params['a']*self.E0)
        else:
            return NaN


    def simulate(self, use_param_vector=None, use_param_vector_name=None, use_times=None, use_current=None):
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

    def set_params_from_vector(self, vector, names):
        for value,name in zip(vector,names):
            self.params[name] = value
            self.dim_params[name] = self.dimensionalise(value,name)

    def test_model(self):
        return """
functions {
    real[] ec_impexp_thomas(real[] ts, real k0, real alpha, real Cdl, real Ru, real E0, real dE, real omega, real Estart) {

        // spatial mesh
        real Xmax = 20.0;
        real dx = 0.02;
        int Nx = 1000; // 20/0.02
        real x[Nx];
        int Nt = size(ts);
        real dt = ts[2]-ts[1];
        real mu = dt/pow(dx,2);

        // input signal
        real E[Nt];

        // solver
        real a = mu;
        real b = 1 + 2*mu;
        real c = mu;
        real d[Nx];
        real e[Nx];
        real f[Nx];
        real U[Nx];
        real V[Nx];
        real Itot[Nt];
        real b0;
        real d0;

        // set up spatial mesh
        for (i in 1:Nx) {
            x[i] = dx*(i-1);
        }

        // set up input signal
        for (i in 1:Nt) {
            E[i] = Estart + ts[i] + dE*sin(omega*ts[i]);
        }

        // setup solver
        for (i in 1:Nx) {
            e[i] = 0.0;
            f[i] = 0.0;
            U[i] = 1.0;
            V[i] = 0.0;
        }
        f[Nx] = 1;
        for (u in 1:Nx-1) {
            int i = Nx-u;
            e[i] = a/(b-c*e[i+1]);
        }

        Itot[1] = 0.0;
        //Itot(1)=Cdl+Cdl*dE*omega/(1+(omega*Ru*Cdl)^2);

        for (n in 1:Nt-1) {
            d = U;
            for (u in 1:Nx-1) {
                int i = Nx-u;
                f[i] = (d[i+1]+f[i+1]*c)/(b-c*e[i+1]);
            }
            b0 = 1+dx*k0*(exp((1-alpha)*(E[n]-E0-Itot[n]*Ru))+exp(-alpha*(E[n]-E0-Itot[n]*Ru)));
            d0 = dx*k0*exp(-alpha*(E[n]-E0-Itot[n]*Ru));
            U[1] = (d0+f[1])/(b0-e[1]);
            for (i in 1:Nx-1) {
                U[i+1] = f[i]+e[i]*U[i];
            }
            Itot[n+1] = (dt*(U[2]-U[1])/dx+Cdl*(E[n+1]-E[n])+Ru*Cdl*Itot[n])/(dt+Ru*Cdl);
        }
        return Itot;
    }
}
data {
    int<lower=1> T;
    real ts[T];
    real k0;
    real alpha;
    real Cdl;
    real Ru;
    real E0;
    real dE;
    real omega;
    real Estart;
}
model {
}
generated quantities {
    real Itot[T];
    print("SAMPLING:");
    Itot = ec_impexp_thomas(ts,k0,alpha,Cdl,Ru,E0,dE,omega,Estart);
}
"""


    def get_stan_model(self):
        return pystan.StanModel(model_name='ECmodel',model_code=self.test_model());

    @property
    def nondim_params(self):
        return self._nondim_params;

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

        if self.dim_params['reversed']:
            I_0 = -D*F*a*c_inf/L_0
        else:
            I_0 = D*F*a*c_inf/L_0

        return E_0,T_0,L_0,I_0


