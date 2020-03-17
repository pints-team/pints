# Examples

This page contains a number of examples showing how to use Pints.

Each example was created as a _Jupyter notebook_ (http://jupyter.org/).
These notebooks can be downloaded and used, or you can simply copy/paste the
relevant code.


## Getting started
- [Optimisation: First example](./optimisation/first-example.ipynb)
- [Sampling: First example](./sampling/first-example.ipynb)
- [Writing a model](./stats/custom-model.ipynb)
- [Writing a custom LogPDF](./stats/custom-logpdf.ipynb)
- [Writing a custom LogPrior](./stats/custom-prior.ipynb)


## Optimisation

### Particle-based methods
- [CMA-ES](./optimisation/cmaes.ipynb)
- [PSO](./optimisation/pso.ipynb)
- [SNES](./optimisation/snes.ipynb)
- [XNES](./optimisation/xnes.ipynb)

### Local optimisers
- [Nelder-Mead](./optimisation/nelder-mead.ipynb)

### Further optimisation
- [Ask-and-tell interface](./optimisation/ask-and-tell.ipynb)
- [Convenience methods fmin() and curve\_fit()](./optimisation/convenience.ipynb)
- [Maximum loglikelihood](./optimisation/maximum-likelihood.ipynb)
- [Multiple objectives](./optimisation/multi-objective.ipynb)
- [Transformed parameter space](./optimisation/transformed-parameters.ipynb)


## Sampling

### MCMC without gradients
- [Differential Evolution MCMC](./sampling/differential-evolution-mcmc.ipynb)
- [DREAM MCMC](./sampling/dream-mcmc.ipynb)
- [Emcee Hammer](./sampling/emcee-hammer.ipynb)
- [Haario Adaptive Covariance MCMC](./sampling/adaptive-covariance-haario.ipynb)
- [Haario-Bardenet Adaptive Covariance MCMC](./sampling/adaptive-covariance-haario-bardenet.ipynb)
- [Metropolis Random Walk MCMC](./sampling/metropolis-mcmc.ipynb)
- [Population MCMC](./sampling/population-mcmc.ipynb)
- [Rao-Blackwell Adaptive Covariance MCMC](./sampling/adaptive-covariance-rao-blackwell.ipynb)
- [Slice Sampling: Doubling MCMC](./sampling/slice-doubling-mcmc.ipynb)
- [Slice Sampling: Overrelaxation MCMC](./sampling/slice-overrelaxation-mcmc.ipynb)
- [Slice Sampling: Stepout MCMC](./sampling/slice-stepout-mcmc.ipynb)

### MCMC with gradients
- [Hamiltonian MCMC](./sampling/hamiltonian-mcmc.ipynb)
- [MALA MCMC](./sampling/mala-mcmc.ipynb)
- [Monomial-Gamma HMC MCMC](./sampling/monomial-gamma-hmc.ipynb)
- [Relativistic MCMC](./sampling/relativistic-mcmc.ipynb)

### Nested sampling
- [Ellipsoidal nested sampling](./sampling/nested-ellipsoidal-sampling.ipynb)
- [Rejection nested sampling](./sampling/nested-rejection-sampling.ipynb)

### Analysing sampling results
- [Autocorrelation](./plotting/mcmc-autocorrelation.ipynb)
- [Effective sample size](./sampling/effective-sample-size.ipynb)
- [Evaluating noise models using autocorrelation plots of the residuals](./plotting/residuals-autocorrelation.ipynb)
- [Pairwise scatterplots](./plotting/mcmc-pairwise-scatterplots.ipynb)
- [Predicted time series](./plotting/mcmc-predicted-time-series.ipynb)
- [Trace plots](./plotting/mcmc-trace-plots.ipynb)


## Statistical modelling
- [Autoregressive moving average errors](./stats/autoregressive-moving-average-errors.ipynb)
- [Cauchy sampling error](./stats/cauchy-sampling-error.ipynb)
- [Integrated noise model](./stats/integrated-gaussian-log-likelihood.ipynb)
- [Log priors](./stats/log-priors.ipynb)
- [Multiplicative Gaussian noise](./stats/multiplicative-gaussian-errors.ipynb)
- [Student-t noise model](./stats/student-t-sampling-error.ipynb)


## Toy problems

### Models
- [Beeler-Reuter action potential model](./toy/model-beeler-reuter-ap.ipynb)
- [Constant model](./toy/model-constant.ipynb)
- [Fitzhugh-Nagumo model](./toy/model-fitzhugh-nagumo.ipynb)
- [Goodwin oscillator model](./toy/model-goodwin-oscillator.ipynb)
- [HES1 Michaelis-Menten model](./toy/model-hes1-michaelis-menten.ipynb)
- [Hodgkin-Huxley Potassium current model](./toy/model-hodgkin-huxley-ik.ipynb)
- [Logistic growth model](./toy/model-logistic.ipynb)
- [Lotka-Volterra predator-prey model](./toy/model-lotka-volterra.ipynb)
- [Repressilator model](./toy/model-repressilator.ipynb)
- [Simple Harmonic Oscillator model](./toy/model-simple-harmonic-oscillator.ipynb)
- [SIR Epidemiology model](./toy/model-sir.ipynb)
- [Stochastic Degradation model](./toy/model-stochastic-degradation.ipynb)

### Distributions
- [Annulus](./toy/distribution-annulus.ipynb)
- [Cone](./toy/distribution-cone.ipynb)
- [German credit hierarchical logistic model](./toy/distribution-german-credit-hierarchical.ipynb)
- [German credit logistic model](./toy/distribution-german-credit.ipynb)
- [High dimensional gaussian](./toy/distribution-high-dimensional-gaussian.ipynb)
- [Multimodal gaussian distribution](./toy/distribution-multimodal-gaussian.ipynb)
- [Neals Funnel](./toy/distribution-neals-funnel.ipynb)
- [Rosenbrock function](./toy/distribution-rosenbrock.ipynb)
- [Simple Egg Box](./toy/distribution-simple-egg-box.ipynb)
- [Twisted Gaussian Banana](./toy/distribution-twisted-gaussian.ipynb)


## Miscellaneous
- [Automatic differentiation using autograd](./toy/automatic-differentiation-using-autograd.ipynb)
- [The example shown on the landing page](./sampling/readme-example.ipynb)
