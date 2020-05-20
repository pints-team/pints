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


Autoregressive order 1
**********************

.. autofunction:: ar1

Autoregressive order 1 with mean 1
**********************************

.. autofunction:: ar1_unity

Autoregressive moving average
*****************************

.. autofunction:: arma11

Autoregressive moving average with mean 1
*****************************************

.. autofunction:: arma11_unity

Independent Gaussian
********************

.. autofunction:: independent

Multiplicative (heteroscedastic) Gaussian
*****************************************

.. autofunction:: multiplicative_gaussian
