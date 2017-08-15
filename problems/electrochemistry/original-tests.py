import matplotlib as mpl
#mpl.use('Agg') # Set this in ~/.config/matplotlib/matplotlibrc if you need it!
import pylab as plt

# The CMA module touches matplotlib: this is temporary fix!
# See: https://github.com/CMA-ES/pycma/issues/17
matplotlib_is_interactive = plt.isinteractive()
import cma
if not matplotlib_is_interactive:
    plt.ioff()

import pints
import electrochemistry
import pickle

import numpy as np
from math import sqrt
import pystan
from math import pi

import seaborn as sns



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

    model = electrochemistry.ECModel(dim_params)

    data = electrochemistry.ECTimeData(filename,model,ignore_begin_samples=40)

    # specify bounds for parameters
    prior = pints.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',pints.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',pints.Uniform(),0,10000)
    prior.add_parameter('Cdl',pints.Uniform(),0,20)
    prior.add_parameter('omega',pints.Uniform(),model.params['omega']*0.99,model.params['omega']*1.01)
    prior.add_parameter('phase',pints.Uniform(),model.params['phase']-pi/10,model.params['phase']+pi/10)

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

    model = electrochemistry.ECModel(dim_params)

    data = electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5,ignore_end_samples=0)

    # specify bounds for parameters
    prior = pints.Prior()
    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    prior.add_parameter('E0',pints.Uniform(),model.params['Estart']+e0_buffer,model.params['Ereverse']-e0_buffer)
    prior.add_parameter('k0',pints.Uniform(),0,10000)
    prior.add_parameter('Cdl',pints.Uniform(),0,20)
    prior.add_parameter('alpha',pints.Uniform(),0.4,0.6)
    prior.add_parameter('Ru',pints.Uniform(),0.0,0.1)

    print 'before cmaes, parameters are:'
    names = prior.get_parameter_names()
    for name in prior.get_parameter_names():
        print name,': ',model.params[name]

    #model.set_params_from_vector([0.00312014718956,2.04189332425,7.274953392],['Cdl','k0','E0'])
    #pints.fit_model_with_cmaes(data,model,prior)

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

    samples = pints.mcmc_with_adaptive_covariance(data,model,prior,nchains=4)
    #samples = np.random.uniform(size=(1000,prior.n+1))

    pickle.dump( samples, open( "samples.p", "wb" ) )
    #samples = pickle.load( open( "samples.p", "rb" ) )

    fig,axes = pints.plot_trace(samples, model, prior)
    plt.savefig('test_mcmc_trace.pdf')
    fig,axes = pints.scatter_grid(samples, model, prior)
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
    #    model = electrochemistry.ECModel(dim_params)
    #    data = electrochemistry.ECTimeData(filename,model,ignore_begin_samples=5)
    #    prior = pints.Prior()
    #    e0_buffer = 0.1*(model.params['Ereverse'] - model.params['Estart'])
    #    prior.add_parameter('E0',pints.Uniform(),
    #                    model.params['Estart']+e0_buffer,
    #                    model.params['Ereverse']-e0_buffer)
    #    prior.add_parameter('k0',pints.Uniform(),0,10000)
    #    prior.add_parameter('Cdl',pints.Uniform(),0,20)
    #    prior.add_parameter('alpha',pints.Uniform(),0.4,0.6)
    #    prior.add_parameter('Ru',pints.Uniform(),0.0,0.1)

    #    pints.fit_model_with_cmaes(data,model,prior)

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


    #samples,theta_samples = pints.hierarchical_gibbs_sampler(priors[0].get_parameter_names(),datas,models,priors)

    #pickle.dump( (samples,theta_samples), open( "hmcmc_done.p", "wb" ) )
    samples,theta_samples = pickle.load( open( "hmcmc_done.p", "rb" ) )
    #theta_samples = pickle.load(open( "hmcmc.p", "rb" ))
    #for i,data,model,prior,theta_sample in zip(range(len(datas)),datas,models,priors,theta_samples):
    #    print 'plotting samples mcmc',i
    #    fig,axes = pints.plot_trace(theta_sample, model, prior)
    #    plt.savefig('mcmc_trace_%d.pdf'%i)
    #    plt.close(fig)

    #for i,data,model,prior,theta_sample in zip(range(len(datas)),datas,models,priors,theta_samples):
    #    print 'plotting samples mcmc',i
    #    fig,axes = pints.scatter_grid(theta_sample, model, prior)
    #    plt.savefig('mcmc_scatter_grid%d.pdf'%i)
    #    plt.close(fig)



    print 'plotting heirarchical samples mcmc diagonal'

    n_param = theta_samples[0].shape[1]
    fig,axes = pints.scatter_diagonal(samples, models[0], priors[0], fig_size=(6, 4*n_param))
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
    fig,axes = pints.plot_trace(samples, models[0], priors[0])
    plt.savefig('test_hmcmc_trace.pdf')

    print 'plotting heirarchical samples mcmc grid'
    fig,axes = pints.scatter_grid(samples, models[0], priors[0])
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
    test_ec_model()
    
    
