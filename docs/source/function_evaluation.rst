*******************
Function evaluation
*******************

.. currentmodule:: pints

The :class:`Evaluator` classes provide an abstraction layer that makes it
easier to implement sequential and/or parallel evaluation of functions.

Example::

    f = pints.SumOfSquaresError(problem)
    e = pints.ParallelEvaluator(f)
    x = [[1, 2],
         [3, 4],
         [5, 6],
         [7, 8],
        ]
    fx = e.evaluate(x)

Overview:

- :func:`evaluate`
- :class:`Evaluator`
- :class:`ParallelEvaluator`
- :class:`SequentialEvaluator`
- :class:`MultiSequentialEvaluator`


.. autofunction:: evaluate

.. autoclass:: Evaluator

.. autoclass:: ParallelEvaluator

.. autoclass:: SequentialEvaluator

.. autoclass:: MultiSequentialEvaluator
