***************
Transformations
***************

.. currentmodule:: pints

:class:`Transform` are objects that provide some convenience parameter
transformation from the model parameter space to a search space. It can be
passed to the :class:`OptimisationController` and :class:`MCMCController` when
doing optimisation and inference.

Parameter transformation is important in many situations, for example
transforming from a constrainted parameter space to some unconstrained variable
search space using :class:`RectangularBoundariesTransform`. Once the
`Transform` object is defined and passed to a controller, all the
transformation work is done behind the scenes by the controller.

Example::

    transform = pints.LogTransform(n_parameters)
    mcmc = pints.MCMCController(log_posterior, n_chains, x0, transform=transform)

Overview:

- :class:`Transform`
- :class:`ComposedTransform`
- :class:`IdentityTransform`
- :class:`LogTransform`
- :class:`LogitTransform`
- :class:`RectangularBoundariesTransform`
- :class:`TransformedLogPDF`
- :class:`TransformedErrorMeasure`
- :class:`TransformedBoundaries`


.. autoclass:: Transform

.. autoclass:: ComposedTransform

.. autoclass:: IdentityTransform

.. autoclass:: LogTransform

.. autoclass:: LogitTransform

.. autoclass:: RectangularBoundariesTransform

.. autoclass:: TransformedLogPDF

.. autoclass:: TransformedErrorMeasure

.. autoclass:: TransformedBoundaries
