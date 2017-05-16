

class Data:
    def __init__(self,filename,params):
        if isinstance(filename,dict):
            self.dim_params = filename
        else:
            key = 'data'+filename.split('data')[1]
            self.dim_params = copy.copy(all_params[key])


        self.dim_params = handle_reversed_scanrate(self.dim_params)
        self.params = e_non_dim_params(self.dim_params)
        if get_data:
            print 'loading data from filename = ',filename,' ...'
            self.exp_data = np.loadtxt(filename,skiprows=19)
            if filename[-11:] == '_cv_current':
                self.exp_t = self.exp_data[:,0]
                self.exp_I = self.exp_data[:,1]
            else:
                self.exp_t = self.exp_data[:,2]
                self.exp_I = self.exp_data[:,1]
            print 'done loading data.'

            self.exp_t,self.exp_I = clean_and_downsample_data(self.exp_t,self.exp_I,self.dim_params,one_period=one_period)
            self.exp_t,self.exp_I = e_non_dim_data(self.exp_t,self.exp_I,self.dim_params)

            maxI = np.max(self.exp_I)
            maxCdl = maxI/(self.params['dE']*self.params['omega'])
            self.params['Cdl_max'] = maxCdl
            print 'found that Cdl (non-dim) must be less than ',maxCdl

            self.t,self.Itot = get_init_sim_data(self.exp_t)
            if simulate:
                e_implicit_exponential_mesh(self.params,self.Itot,self.t)
                self.exp_I = np.array(self.Itot)
                self.exp_t = np.array(self.t)

            if do_calculate_harmonics:
                self.hanning = np.hanning(len(self.exp_I))
                print 'calculating harmonics 1...'
                i = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
                w = np.array([1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ])
                self.freq_weights,self.F_exp_I_filtered = calculate_harmonics(self.exp_t,self.exp_I,self.params['omega'],i,w,rescale_harmonics=False,include_noise=False)

                print 'calculating harmonics 2...'
                #i = np.array([4,5,6,7,8,9,10,11,12])
                i = np.array([4,5,6,7,8])
                #w = np.array([1,1,1,1,1,1,1 ,1 ,1 ])
                w = np.array([1,1,1,1,1])
                self.freq_weights2,self.F_exp_I_filtered2 = calculate_harmonics(self.exp_t,self.exp_I,self.params['omega'],i,w,rescale_harmonics=False,include_noise=False)

                #print 'calculating harmonics 3...'
                #i = np.array([5,6,7])
                #w = np.array([1,1,1])
                #self.freq_weights3,self.F_exp_I_filtered3 = calculate_harmonics(self.exp_t,self.exp_I,self.params['omega'],i,w,include_noise=False)

                #print 'done calculating harmonics.'
            if do_Edata:
                print 'calculating Edc_data...'
                self.Edc_data = vector()
                self.dEdc_datadt = vector()
                w,F = calculate_harmonics(self.exp_t,self.exp_I,self.params['omega'],np.array([0]),np.array([1]),include_noise=False,rescale_harmonics=False)
                f_dc_numpy = fft.irfft(fft.rfft(self.exp_I)*w)
                f_dc_vector = vector()
                for val in f_dc_numpy:
                    f_dc_vector.append(val)
                calc_E_data(self.params,f_dc_vector,self.Edc_data,self.dEdc_datadt,self.t)
                print 'done calculating Edc_data.'


        else:
            self.Itot = vector()
            self.t = vector()
        if self.dim_params['dispersion'] == True:
            if self.dim_params['problem'] == 'E':
                self.interpolant = create_interpolant_E1(self,interpolant_l)
            elif self.dim_params['problem'] == 'E2':
                self.interpolant = create_interpolant_E2(self,interpolant_l)


