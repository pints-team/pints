#
# Plots predicted posterior time series
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import numpy as np


def series(samples, problem, ref_parameters=None, thinning=None):
    """
    Creates and returns a plot of predicted time series for a given list of
    ``samples`` and a single-output or multi-output ``problem``.

    Because this method runs simulations, it can take a considerable time to
    run.

    Returns a ``matplotlib`` figure object and axes handle.

    Parameters
    ----------
    samples
        A list of samples, with shape ``(n_samples, n_parameters)``, where
        `n_samples` is the number of samples in the list and ``n_parameters``
        is the number of parameters.
    problem
        A :class:``pints.SingleOutputProblem`` or
        :class:``pints.MultiOutputProblem`` of a n_parameters equal to or
        greater than the ``n_parameters`` of the `samples`. Any extra
        parameters present in the chain but not accepted by the
        ``SingleOutputProblem`` or ``MultiOutputProblem`` (for example
        parameters added by a noise model) will be ignored.
    ref_parameters
        A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    thinning
        An integer exceeding zero. If specified, only every
        n-th sample (with ``n = thinning``) in the samples will be used. If
        left at the default value ``None``, a value will be chosen so that
        200 to 400 predictions are shown.
    """
    import matplotlib.pyplot as plt

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample,'
                         + ' n_parameters).')

    # Get problem n_parameters
    n_parameters = problem.n_parameters()

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param and \
                len(ref_parameters) != n_parameters:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')
        ref_series = problem.evaluate(ref_parameters[:n_parameters])

    # Get number of problem output
    n_outputs = problem.n_outputs()

    # Get thinning rate
    if thinning is None:
        thinning = max(1, int(n_sample / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError(
                'Thinning rate must be `None` or an integer greater than'
                ' zero.')

    # Get times
    times = problem.times()

    # Evaluate the model for all parameter sets in the samples
    i = 0
    predicted_values = []
    for params in samples[::thinning, :n_parameters]:
        predicted_values.append(problem.evaluate(params))
        i += 1
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)

    # Guess appropriate alpha (0.05 worked for 1000 plots)
    alpha = min(1, max(0.05 * (1000 / (n_sample / thinning)), 0.5))

    # Plot prediction
    fig, axes = plt.subplots(n_outputs, 1, figsize=(8, np.sqrt(n_outputs) * 3),
                             sharex=True)

    if n_outputs == 1:
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.plot(
            times, problem.values(), 'x', color='#7f7f7f', ms=6.5, alpha=0.5,
            label='Original data')
        plt.plot(
            times, predicted_values[0], color='#1f77b4',
            label='Inferred series')
        for v in predicted_values[1:]:
            plt.plot(times, v, color='#1f77b4', alpha=alpha)
        plt.plot(times, mean_values, 'k:', lw=2,
                 label='Mean of inferred series')

        # Add reference series if given
        if ref_parameters is not None:
            plt.plot(times, ref_series, color='#d62728', ls='--',
                     label='Reference series')

        plt.legend()

    elif n_outputs > 1:
        # Remove horizontal space between axes and set common xlabel
        fig.subplots_adjust(hspace=0)
        axes[-1].set_xlabel('Time')

        # Go through each output
        for i_output in range(n_outputs):
            axes[i_output].set_ylabel('Output %d' % (i_output + 1))
            axes[i_output].plot(
                times, problem.values()[:, i_output], 'x', color='#7f7f7f',
                ms=6.5, alpha=0.5, label='Original data')
            axes[i_output].plot(
                times, predicted_values[0][:, i_output], color='#1f77b4',
                label='Inferred series')
            for v in predicted_values[1:]:
                axes[i_output].plot(times, v[:, i_output], color='#1f77b4',
                                    alpha=alpha)
            axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2,
                                label='Mean of inferred series')

            # Add reference series if given
            if ref_parameters is not None:
                axes[i_output].plot(times, ref_series[:, i_output],
                                    color='#d62728', ls='--',
                                    label='Reference series')

        axes[0].legend()

    plt.tight_layout()
    return fig, axes
