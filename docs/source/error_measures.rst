**************
Error measures
**************

.. currentmodule:: pints

Error measures are callable objects that return some scalar representing the
error between a model and an experiment.

Example::

    error = pints.SumOfSquaresError(problem)
    x = [1,2,3]
    fx = error(x)

Overview:

- :class:`ErrorMeasure`
- :class:`MeanSquaredError`
- :class:`NormalisedRootMeanSquaredError`
- :class:`ProbabilityBasedError`
- :class:`ProblemErrorMeasure`
- :class:`RootMeanSquaredError`
- :class:`SumOfErrors`
- :class:`SumOfSquaresError`


Customisable error measure
**************************

.. autoclass:: ErrorMeasure

Mean squared error
******************

.. autoclass:: MeanSquaredError

Normalised root mean squared error
**********************************

.. autoclass:: NormalisedRootMeanSquaredError

Probability-based error
***********************

.. autoclass:: ProbabilityBasedError

Problem-based error
*******************

.. autoclass:: ProblemErrorMeasure

Root mean squared error
***********************

.. autoclass:: RootMeanSquaredError

Sum of errors
*************

.. autoclass:: SumOfErrors

Sum of squares error
********************

.. autoclass:: SumOfSquaresError

