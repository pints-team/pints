import hobo
import hobo.electrochemistry
import pickle

import numpy as np
from math import sqrt
import pystan
import cma
from math import pi

import matplotlib as mpl
mpl.use('Agg')
import pylab as plt

import seaborn as sns

def test_ec_model():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

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
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.ECModel(dim_params)
    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # calculate model at time points given by the data file
    I,t = model.simulate(use_times=data.time)

    plt.figure()
    plt.plot(t,I)
    plt.plot(data.time,data.current)
    plt.show(block=True)

    error = data.distance(I)
    print 'error = ',error
    assert 0.07 > error

def test_pom_model():
    filename = '../POMGCL_6020104_1.0M_SFR_d16.txt'
    dim_params = {
        'reversed': False,
        'Estart': 0.6,
        'Ereverse': -0.1,
        'omega': 60.05168,
        'phase': 0,
        'dE': 20e-3,
        'v': -0.1043081,
        't_0': 0.00,
        'T': 298.2,
        'a': 0.0707,
        'c_inf': 0.1*1e-3*1e-3,
        'Ru': 50.0,
        'Cdl': 0.000008,
        'alpha1': 0.5,
        'alpha2': 0.5,
        'Gamma' : 0.7*53.0e-12,
        'E01': 0.368,
        'E02': 0.338,
        'E11': 0.227,
        'E12': 0.227,
        'E21': 0.011,
        'E22': -0.016,
        'k01': 7300,
        'k02': 7300,
        'k11': 1e4,
        'k12': 1e4,
        'k21': 2500,
        'k22': 2500
        }

    model = hobo.electrochemistry.POMModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # calculate model at time points given by the data file
    I,t = model.simulate(use_times=data.time)

    plt.figure()
    plt.plot(t,I)
    plt.plot(data.time,data.current)
    plt.show(block=True)


def test_nonlin_op():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

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
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.ECModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',hobo.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    prior.add_parameter('Cdl',hobo.Uniform(),0,20)

    params = hobo.fit_model_with_cmaes(data,model,prior)
    print params

    model.set_params_from_vector(params,prior.get_parameter_names())
    I,t = model.simulate(use_times=data.time)

    assert data.distance(I) < 0.07


def test_stan():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

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
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.ECModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',hobo.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    prior.add_parameter('Cdl',hobo.Uniform(),0,20)
    prior.add_parameter('omega',hobo.Uniform(),model.params['omega']*0.99,model.params['omega']*1.01)
    prior.add_parameter('phase',hobo.Uniform(),model.params['phase']-pi/10,model.params['phase']+pi/10)

    # calculate model using stan
    sm = model.get_stan_model()

    combined_params = dict(T=len(data.time),ts=data.time)
    combined_params.update(model.nondim_params)
    op = sm.sampling(data=combined_params, chains=1, iter=1, warmup=0,algorithm='Fixed_param')
    print 'finished'

    stan_current = op.extract('Itot')

    f = plt.figure()
    plt.plot(data.time,data.current,label='exp')
    plt.plot(data.time,stan_current,label='stan')
    plt.show()

def test_mcmc():
    filename = '../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt'

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
        'Cdl': 20.0*1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53
        }

    model = hobo.electrochemistry.ECModel(dim_params)

    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5,ignore_end_samples=0)

    # specify bounds for parameters
    prior = hobo.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',hobo.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    prior.add_parameter('Cdl',hobo.Uniform(),0,20)
    prior.add_parameter('alpha',hobo.Uniform(),0.4,0.6)
    prior.add_parameter('Ru',hobo.Uniform(),0.0,0.1)

    print 'before cmaes, parameters are:'
    names = prior.get_parameter_names()
    for name in prior.get_parameter_names():
        print name,': ',model.params[name]

    #model.set_params_from_vector([0.00312014718956,2.04189332425,7.274953392],['Cdl','k0','E0'])
    #hobo.fit_model_with_cmaes(data,model,prior)

    #pickle.dump( (data,model,prior), open( "test_mcmc.p", "wb" ) )
    data,model,prior = pickle.load(open( "test_mcmc.p", "rb" ))

    I,t = model.simulate(use_times=data.time)
    plt.figure()
    plt.plot(t,I)
    plt.plot(data.time,data.current)
    plt.savefig('test_mcmc_after_cmaes.pdf')
    plt.close()

    print 'after cmaes, parameters are:'
    names = prior.get_parameter_names()
    for name in prior.get_parameter_names():
        print name,': ',model.params[name]

    samples = hobo.mcmc_with_adaptive_covariance(data,model,prior,nchains=4)
    #samples = np.random.uniform(size=(1000,prior.n+1))

    pickle.dump( samples, open( "samples.p", "wb" ) )
    #samples = pickle.load( open( "samples.p", "rb" ) )

    fig,axes = hobo.plot_trace(samples, model, prior)
    plt.savefig('test_mcmc_trace.pdf')
    fig,axes = hobo.scatter_grid(samples, model, prior)
    plt.savefig('test_mcmc_after_mcmc.pdf')
    #plt.show(block=True)

def test_hierarchical():
    #filenames = ['../GC01_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt',
    #             '../GC03_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC04_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC05_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC06_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC07_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC08_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC09_FeIII-1mM_1M-KCl_02_009Hz.txt',
    #             '../GC10_FeIII-1mM_1M-KCl_02_009Hz.txt']

    #dim_params = {
    #    'reversed': True,
    #    'Estart': 0.5,
    #    'Ereverse': -0.1,
    #    'omega': 9.0152,
    #    'phase': 0,
    #    'dE': 0.08,
    #    'v': -0.08941,
    #    't_0': 0.001,
    #    'T': 297.0,
    #    'a': 0.07,
    #    'c_inf': 1*1e-3*1e-3,
    #    'D': 7.2e-6,
    #    'Ru': 8.0,
    #    'Cdl': 20.0*1e-6,
    #    'E0': 0.214,
    #    'k0': 0.0101,
    #    'alpha': 0.53
    #    }


    #datas = []
    #models = []
    #priors = []
    #for filename in filenames:
    #    model = hobo.electrochemistry.ECModel(dim_params)
    #    data = hobo.electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5)
    #    prior = hobo.Prior()
    #    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    #    prior.add_parameter('E0',hobo.Uniform(),
    #                    model.params['Estart']+e0_buffer,
    #                    model.params['Ereverse']-e0_buffer)
    #    prior.add_parameter('k0',hobo.Uniform(),0,10000)
    #    prior.add_parameter('Cdl',hobo.Uniform(),0,20)
    #    prior.add_parameter('alpha',hobo.Uniform(),0.4,0.6)
    #    prior.add_parameter('Ru',hobo.Uniform(),0.0,0.1)

    #    hobo.fit_model_with_cmaes(data,model,prior)

    #    datas.append(data)
    #    models.append(model)
    #    priors.append(prior)

    #pickle.dump( (datas,models,priors), open( "test_hierarchical.p", "wb" ) )
    datas,models,priors = pickle.load(open( "test_hierarchical.p", "rb" ))
    #plt.figure()
    #for i,data,model,prior in zip(range(len(datas)),datas,models,priors):
    #    print 'plotting cmaes fit',i
    #    plt.clf()
    #    I,t = model.simulate(use_times=data.time)
    #    plt.plot(t,I,label='sim')
    #    plt.plot(data.time,data.current,label='exp')
    #    plt.savefig('cmaes_fit_%d.pdf'%i)


    #samples,theta_samples = hobo.hierarchical_gibbs_sampler(priors[0].get_parameter_names(),datas,models,priors)

    #pickle.dump( (samples,theta_samples), open( "hmcmc_done.p", "wb" ) )
    samples,theta_samples = pickle.load( open( "hmcmc_done.p", "rb" ) )
    #theta_samples = pickle.load(open( "hmcmc.p", "rb" ))
    #for i,data,model,prior,theta_sample in zip(range(len(datas)),datas,models,priors,theta_samples):
    #    print 'plotting samples mcmc',i
    #    fig,axes = hobo.plot_trace(theta_sample, model, prior)
    #    plt.savefig('mcmc_trace_%d.pdf'%i)
    #    plt.close(fig)

    #for i,data,model,prior,theta_sample in zip(range(len(datas)),datas,models,priors,theta_samples):
    #    print 'plotting samples mcmc',i
    #    fig,axes = hobo.scatter_grid(theta_sample, model, prior)
    #    plt.savefig('mcmc_scatter_grid%d.pdf'%i)
    #    plt.close(fig)



    print 'plotting heirarchical samples mcmc diagonal'

    n_param = theta_samples[0].shape[1]
    fig,axes = hobo.scatter_diagonal(samples, models[0], priors[0], fig_size=(6, 4*n_param))
    for theta_sample_i in theta_samples:
        print 'plotting sample'
        for i in range(n_param):
            sns.kdeplot(theta_sample_i[:, i], ax=axes[i].twinx(),shade=False)
    #axes = [[line.get_ydata()/np.amax(line.get_ydata()) for line in ax.get_lines()] for ax in axes]

    #for i in range(n_param):
    #    lines = axes[i].get_lines()
    #    for j,line in enumerate(lines):
    #        axes[i].lines[j].set_ydata(line.get_ydata()/np.amax(line.get_ydata()))

    plt.savefig('hmcmc_diagonal.pdf')

    print 'plotting heirarchical samples mcmc trace'
    fig,axes = hobo.plot_trace(samples, models[0], priors[0])
    plt.savefig('test_hmcmc_trace.pdf')

    print 'plotting heirarchical samples mcmc grid'
    fig,axes = hobo.scatter_grid(samples, models[0], priors[0])
    for theta_sample_i in theta_samples:
        for i in range(n_param):
            print i,prior.get_parameter_names()[i],': (min,max) = (',np.min(theta_sample_i[:,i]),',',np.max(theta_sample_i[:,i]),')'
            for j in range(n_param):
                if i == j:
                    print 'plotting'
                    sns.kdeplot(theta_sample_i[:, i], ax=axes[i, j])

    plt.savefig('hmcmc.pdf')
    plt.close(fig)
    #plt.show(block=True)




if __name__ == "__main__":
    test_mcmc()
