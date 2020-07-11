***************
Transformations
***************

.. currentmodule:: pints

:class:`Transformation` objects provide methods to transform between different
representations of a parameter space; for example from a "model space" where
parameters have units and some physical counterpart to a "search space" where
parameters are non-dimensionalised and less-recognisable to the modeller.
The transformed space may in many cases prove simpler to work with for
inference: leading to more effective and efficient optimisation and sampling.

To perform optimisation or sampling in a transformed space, users can choose to
write their :class:`pints.ForwardModel` in "search space" directly, but the
issue with this is that we will not be correctly inferring the "model
parameters". An alternative is to write the ``ForwardModel`` in model
parameters, and pass a :class:`Transformation` object to e.g. an
:class:`OptimisationController` or :class:`MCMCController`. Using the
:class:`Transformation` object ensures users get the correct statistics about
the model parameters (not the search space parameters).

Parameter transformation can be useful in many situations, for example
transforming from a constrained parameter space to an unconstrained search
space using :class:`RectangularBoundariesTransformation` leads to crucial
performance improvements for many methods.

Example::

    transform = pints.LogTransformation(n_parameters)
    mcmc = pints.MCMCController(log_posterior, n_chains, x0, transform=transform)

Overview:

- :class:`ComposedTransformation`
- :class:`ComposedElementWiseTransformation`
- :class:`ElementWiseTransformation`
- :class:`IdentityTransformation`
- :class:`LogitTransformation`
- :class:`LogTransformation`
- :class:`RectangularBoundariesTransformation`
- :class:`Transformation`
- :class:`TransformedBoundaries`
- :class:`TransformedErrorMeasure`
- :class:`TransformedLogPDF`
- :class:`TransformedLogPrior`


.. autoclass:: ComposedTransformation

.. autoclass:: ComposedElementWiseTransformation

.. autoclass:: ElementWiseTransformation

.. autoclass:: IdentityTransformation

.. autoclass:: LogitTransformation

.. autoclass:: LogTransformation

.. autoclass:: RectangularBoundariesTransformation

.. autoclass:: Transformation

.. autoclass:: TransformedBoundaries

.. autoclass:: TransformedErrorMeasure

.. autoclass:: TransformedLogPDF

.. autoclass:: TransformedLogPrior
