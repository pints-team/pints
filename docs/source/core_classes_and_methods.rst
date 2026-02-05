************************
Core classes and methods
************************

.. currentmodule:: pints

Pints provides the :class:`SingleOutputProblem`,
:class:`MultiOutputProblem`, :class:`ProblemCollection`
and :class:`SubProblem` classes to formulate
inverse problems based on time series data and
:class:`ForwardModel`.

Overview:

- :class:`ForwardModel`
- :class:`ForwardModelS1`
- :class:`MultiOutputProblem`
- :class:`ProblemCollection`
- :class:`SingleOutputProblem`
- :class:`SubProblem`
- :class:`TunableMethod`
- :func:`version`

.. autofunction:: version

.. autoclass:: TunableMethod


Forward model
*************

.. autoclass:: ForwardModel

Forward model with sensitivities
********************************

.. autoclass:: ForwardModelS1

Problems
********

.. autoclass:: MultiOutputProblem

.. autoclass:: ProblemCollection

.. autoclass:: SingleOutputProblem

.. autoclass:: SubProblem
