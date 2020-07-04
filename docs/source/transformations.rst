***************
Transformations
***************

.. currentmodule:: pints

:class:`Transform` objects provide methods to transform between different
representations of a parameter space; for example from a "model space" where
parameters have units and some physical counterpart to a "search space" where
parameters are non-dimensionalised and less-recognisable to the modeller but
easier to deal with mathematically.

To perform optimisation or sampling in a transformed space, users can choose to
write their :class:`pints.ForwardModel` in "search space" directly.
But an alternative is to write the ``ForwardModel`` in model parameters, and
pass a :class:`Transform` object to e.g. an :class:`OptimisationController` or :class:`MCMCController`.

Parameter transformation can be useful in many situations, for example
transforming from a constrained parameter space to an unconstrained search
space using :class:`RectangularBoundariesTransform` leads to crucial
performance improvements for many methods.

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
