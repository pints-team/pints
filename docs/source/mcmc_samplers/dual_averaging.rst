**************
Dual Averaging
**************

Dual averaging is not a sampling method, but is a method of adaptivly tuning the 
Hamintonian Monte Carlo (HMC) step size and mass matrix for the particular log-posterior 
being sampled. Pint's NUTS sampler uses dual averaging, but we have defined the dual 
averaging method separately so that in the future it can be used in HMC and other 
HMC-derived samplers.


.. currentmodule:: pints

.. autoclass:: DualAveragingAdaption
