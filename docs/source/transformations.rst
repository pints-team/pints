***************
Transformations
***************

.. currentmodule:: pints

:class:`Transformation` objects provide methods to transform between different
representations of a parameter space; for example from a "model space"
(:math:`p`) where parameters have units and some physical counterpart to
a "search space" (e.g. :math:`q = \log(p)`) where parameters are
non-dimensionalised and less-recognisable to the modeller.
The transformed space may in many cases prove simpler to work with for
inference: leading to more effective and efficient optimisation and sampling.

To perform optimisation or sampling in a transformed space, users can choose to
write their :class:`pints.ForwardModel` in "search space" directly, but the
issue with this is that we will no longer be correctly inferring the "model
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
- :class:`IdentityTransformation`
- :class:`LogitTransformation`
- :class:`LogTransformation`
- :class:`RectangularBoundariesTransformation`
- :class:`ScalingTransformation`
- :class:`Transformation`
- :class:`TransformedBoundaries`
- :class:`TransformedErrorMeasure`
- :class:`TransformedLogPDF`
- :class:`TransformedLogPrior`
- :class:`UnitCubeTransformation`


Transformation types
********************

.. autoclass:: Transformation

.. autoclass:: ComposedTransformation

.. autoclass:: IdentityTransformation

.. autoclass:: LogitTransformation

.. autoclass:: LogTransformation

.. autoclass:: RectangularBoundariesTransformation

.. autoclass:: ScalingTransformation

.. autoclass:: UnitCubeTransformation

Transformed objects
*******************

.. autoclass:: TransformedBoundaries

.. autoclass:: TransformedErrorMeasure

.. autoclass:: TransformedLogPDF

.. autoclass:: TransformedLogPrior

.. autoclass:: TransformedRectangularBoundaries

