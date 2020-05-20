****************
Noise generators
****************

.. module:: pints.noise

Pints contains a module ``pints.noise`` that contains methods that generate
 different kinds of noise.
This can then be added to simulation output to create "realistic" experimental
 data.

 Overview:

 - :func:`ar1`
 - :func:`ar1_unity`
 - :func:`arma11`
 - :func:`arma11_unity`
 - :func:`independent`
 - :func:`multiplicative_gaussian`


Autoregressive order 1 noise
****************************

.. autofunction:: ar1

Autoregressive order 1 with mean 1 noise
****************************************

.. autofunction:: ar1_unity

Autoregressive moving average noise
***********************************

.. autofunction:: arma11

Autoregressive moving average with mean 1 noise
***********************************************

.. autofunction:: arma11_unity

Independent Gaussian noise
**************************

.. autofunction:: independent

Multiplicative (heteroscedastic) Gaussian noise
***********************************************

.. autofunction:: multiplicative_gaussian
