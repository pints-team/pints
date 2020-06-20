***************
Transformations
***************

.. currentmodule:: pints

:class:`Transform` are objects that provide some convenience parameter
transformation from the model parameter space to a search space. It can be
passed to the :class:`OptimisationController` and :class:`MCMCController` when
doing optimisation and inference.

Example::

    transform = pints.LogTransform()
    mcmc = pints.MCMCController(log_posterior, n_chains, x0, transform=transform)

Overview:

- :class:`Transform`
- :class:`LogTransform`
- :class:`LogitTransform`
- :class:`RectangularBoundariesTransform`
- :class:`TransformedLogPDF`
- :class:`TransformedErrorMeasure`
- :class:`TransformedBoundaries`


.. autoclass:: Transform

.. autoclass:: LogTransform

.. autoclass:: LogitTransform

.. autoclass:: RectangularBoundariesTransform

.. autoclass:: TransformedLogPDF

.. autoclass:: TransformedErrorMeasure

.. autoclass:: TransformedBoundaries
