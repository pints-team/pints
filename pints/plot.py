#
# Quick diagnostic tools for inference results
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import numpy as np
from scipy import stats
try:
    import matplotlib.pyplot as plt
except:
    raise ImportError("The module pints.plot requires matplotlib.pyplot")

def trace(chain, *args):
    """
    This method creates and returns a trace plot for the given MCMC chain(s).

    Arguments:

    `chain`
        MCMC routine returned chain.
        More `chain`s can be input follow after the first chain.

    Return:

    `fig`
        A `matplotlib` figure object.

    """
    n_sample, n_param = chain.shape

    fig, axes = plt.subplots(n_param, 2, figsize=(12, 2*n_param))
    for i in xrange(n_param):
        # Add histogram subplot
        axes[i,0].set_xlabel('Parameter ' + str(i + 1))
        axes[i,0].set_ylabel('Frequency')
        axes[i,0].hist(chain[:,i], label='p' + str(i + 1), bins=40, 
                color='green', alpha=0.5)

        # Add trace subplot
        axes[i,1].set_xlabel('Iteration')
        axes[i,1].set_ylabel('Parameter ' + str(i + 1))
        axes[i,1].plot(chain[:,i], color='green', alpha=0.5)

    if args:
        for more_chain in args:
            n_sample, n_param = more_chain.shape
            if n_param != chain.shape[1]:
                raise Exception('Input chains must have the same number of'
                        ' parameters')
            for i in xrange(n_param):
                axes[i,0].hist(more_chain[:,i], bins=40, alpha=0.5)
                axes[i,1].plot(more_chain[:,i], alpha=0.5)            

    plt.tight_layout()
    return fig

def autocorrelation(chain, max_lags=20):
    """
    This method creates and returns an autocorrelation plot for the given MCMC
    chain.

    Arguments:

    `chain`
        MCMC routine returned chain.
    `max_lags`
        (Optional) number of lags to show. Default max_lags=20.

    Return:

    `fig`
        A `matplotlib` figure object.

    """
    n_sample, n_param = chain.shape

    fig, axes = plt.subplots(n_param, 1, sharex=True, figsize=(6, 2*n_param))
    for i in xrange(n_param):
        axes[i].acorr(chain[:,i] - np.mean(chain[:,i]), maxlags=max_lags)
        axes[i].set_xlim(-0.5, max_lags+0.5)
    axes[i].set_xlabel('Lag')
    #fig.text(0.04, 0.5, 'Autocorrelation', va='center', rotation='vertical')
    axes[int(i/2)].set_ylabel('Autocorrelation')

    plt.tight_layout()
    return fig

def prediction(chain, problem):
    """
    This method creates and returns a predicted time series plot based on the
    MCMC reults.

    Arguments:

    `chain`
        MCMC routine returned chain.
    `problem`
        pints.SingleSeriesProblem object.

    Return:
    
    `fig`
        A `matplotlib` figure object.

    """
    if not isinstance(problem, pints.SingleSeriesProblem):
        raise TypeError('Second argument to prediction() must be'
                ' pints.SingleSeriesProblem type')

    n_sample, n_param = chain.shape
    # Aviod having too many lines/simulations
    plot_n_sample = 1000 if n_sample>1000 else n_sample
    times = problem.times()

    # Evaluate the model for all inferred parameters
    predicted_values = []
    for params in chain[-plot_n_sample:,:]:
        predicted_values.append(problem.evaluate(params[:-1]))
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)

    # Plot prediction
    fig = plt.figure(figsize=(7.5, 3.75))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot(times, predicted_values[0], color='#1f77b4', label='inferred series')
    for v in predicted_values[1:]:
        plt.plot(times, v, color='#1f77b4', alpha=0.05)
    plt.plot(times, mean_values, color='black', lw=2, label='mean inferred')
    plt.plot(times, problem.values(), 'o', color='#7f7f7f', ms=6.5, label='data points')

    return fig

def _force_equal_aspect_ratio(ax):
    """
    Force aspect ratio to be equal.

    see: https://stackoverflow.com/questions/7965743

    `ax`
        `matplotlib` axes handle.

    """
    im = ax.get_images()
    ex = im[0].get_extent()
    ax.set_aspect(abs((ex[1]-ex[0])/(ex[3]-ex[2])))


def histogram(axes, X, kde=False):
    """
    This method plots 1D histogram of the given variable to the given 
    `matplotlib` axes.

    Arguments:

    `axes`
        `matplotlib` axes handle.
    `X`
        An array of the variable to plot.
    `kde` (bool)
        If True, use KDE to estimate the density distribution. Default False.

    """
    Xmin = np.min(X)
    Xmax = np.max(X)
    x_hist = np.linspace(Xmin, Xmax, 50)
    axes.hist(X, bins=x_hist, normed=True)
    if kde:
        x_kde = np.linspace(Xmin, Xmax, 100)
        kernel = stats.gaussian_kde(X)
        Z = kernel(x_kde)
        axes.plot(x_kde, Z)
    #_force_equal_aspect_ratio(ax)


def scatter(axes, X, Y, kde=False):
    """
    This method plots 2D pairwise plot of the two given variables to the given
    `matplotlib` axes.

    Arguments:

    `axes`
        `matplotlib` axes handle.
    `X`
        An array of the first variable to plot.
    `Y`
        An array of the second variable to plt.
    `kde` (bool)
        If True, use KDE to estimate the density distribution. Default False.

    """
    Xmin, Xmax = np.min(X), np.max(X)
    Ymin, Ymax = np.min(Y), np.max(Y)
    if not kde:
        x_hist = np.linspace(Xmin, Xmax, 25)
        y_hist = np.linspace(Ymin, Ymax, 25)
        axes.hist2d(X, Y, bins=[x_hist,y_hist], normed=True, cmap=plt.cm.Blues)
    else:
        X_kde, Y_kde = np.mgrid[Xmin:Xmax:100j, Ymin:Ymax:100j]
        positions = np.vstack([X_kde.ravel(), Y_kde.ravel()])
        values = np.vstack([X, Y])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X_kde.shape)
        axes.imshow(np.rot90(Z), cmap=plt.cm.Blues, extent=[Xmin, Xmax, Ymin, Ymax])
        axes.plot(X, Y, 'k.', markersize=2)
    _force_equal_aspect_ratio(axes)



def pairwise_scatter(chain, kde=False):
    """
    This method creates and returns a pairwise scatterplot matrix for the 
    given MCMC chain.

    Arguments:

    `chain`
        MCMC routine returned chain.
    `kde` (bool)
        if True, use KDE to estimate the density distribution. Default False.

    Return:

    `fig`
        A `matplotlib` figure object.

    """
    n_sample, n_param = chain.shape
    fig_size = (3*n_param, 3*n_param)

    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    for i in range(n_param):
        for j in range(n_param):
            if i == j:
                # Plot the diagonal
                histogram(axes[i,j], chain[:,i], kde=kde)
            elif i < j:
                axes[i,j].axis('off')
            else:
                # Plot the samples as density map
                scatter(axes[i,j], chain[:,j], chain[:,i], kde=kde)
            if i < n_param-1:
                # Only show x tick labels for the last row
                axes[i,j].set_xticklabels([])
            else:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i,j].get_xticklabels():
                    tl.set_rotation(45)
            if j > 0:
                # Only show y tick labels for the first column
                axes[i,j].set_yticklabels([])
        if i > 0:
            # The first one is not a parameter
            axes[i,0].set_ylabel('parameter %d'%(i+1))
        else:
            axes[i,0].set_ylabel('probability density')
        axes[-1,i].set_xlabel('parameter %d'%(i+1))

    return fig

