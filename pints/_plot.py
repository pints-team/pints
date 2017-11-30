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
import matplotlib.pyplot as plt

def trace(chain):
    """
    Trace plot.

    `chain`
        MCMC routine returned chain

    """
    n_sample, n_param = chain.shape

    fig = plt.figure(figsize=(14, 9))
    for i in xrange(n_sample):
        # Add histogram subplot
        plt.subplot(n_param, 2, 1 + 2 * i)
        plt.xlabel('Parameter ' + str(i + 1))
        plt.ylabel('Frequency')
        plt.hist(chain[:,i], label='p' + str(i + 1), bins=40, color='green',
        alpha=0.5)

        # Add trace subplot
        plt.subplot(n_param, 2, 2 + 2 * i)
        plt.xlabel('Iteration')
        plt.ylabel('Parameter ' + str(i + 1))
        plt.plot(chain[:,i], color='green', alpha=0.5)

    plt.tight_layout()
    return fig

def autocorrelation(chain):
    """
    Autocorrelation plot for MCMC.

    `chain`
        MCMC routine returned chain

    """
    n_sample, n_param = chain.shape

    fig, axes = plt.subplots(1, n_param, sharex=True, figsize=(6, 2*n_param))
    for i in xrange(n_sample):
        axes[i].acorr(chain[:,i] - np.mean(chain[:,i]), label='p'+str(i+1))
        axes[i].set_xlim(-0.5, 20.5)
    axes[i].set_xlabel('Lag')
    fig.text(0.04, 0.5, 'Autocorrelation', va='center', rotation='vertical')

    return fig

def prediction(chain, problem):
    """
    Predicted time series plot.

    `chain`
        MCMC routine returned chain
    `problem`
        pints.SingleSeriesProblem object

    """
    if not isinstance(problem, pints.SingleSeriesProblem):
        raise TypeError('pints.SingleSeriesProblem type is expected')

    n_sample, n_param = chain.shape
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
    pl.plot(times, predicted_values[0], color='#1f77b4', label='inferred series')
    for v in predicted_values[1:]:
        plt.plot(times, v, color='#1f77b4', alpha=0.05)
    plt.plot(times, mean_values, color='black', lw=2, label='mean inferred')
    plt.plot(times, problem.values(), 'o', color='#7f7f7f', ms=6.5, label='data points')

    return fig

def forceAspect(ax, aspect=1):
    """
    Set aspect.

    `ax`
        matplotlib axes handler
    `aspect`
        aspect to be set

    """
    im = ax.get_images()
    ex = im[0].get_extent()
    ax.set_aspect(abs((ex[1]-ex[0])/(ex[3]-ex[2]))/aspect)

def hist2d(X, Y, ax):
    """
    2D histogram plot.

    `X`
        An array of the first variable
    `Y`
        An array of the second variable
    `ax`
        matplotlib axes handler

    """
    Xmin, Xmax = np.min(X), np.max(X)
    Ymin, Ymax = np.min(Y), np.max(Y)
    x2 = np.linspace(Xmin, Xmax, 25)
    y2 = np.linspace(Ymin, Ymax, 25)
    histplot = ax.hist2d(X, Y, bins=[x2,y2], normed=True, cmap=plt.cm.Blues)
    forceAspect(ax)

def kde2d(X, Y, ax):
    """
    2D kernel density estimation plot.

    `X`
        An array of the first variable
    `Y`
        An array of the second variable
    `ax`
        matplotlib axes handler

    """
    Xmin, Xmax = np.min(X), np.max(X)
    Ymin, Ymax = np.min(Y), np.max(Y)
    X1, Y1 = np.mgrid[Xmin:Xmax:100j, Ymin:Ymax:100j]
    positions = np.vstack([X1.ravel(), Y1.ravel()])
    values = np.vstack([X, Y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X1.shape)
    ax.imshow(np.rot90(Z), cmap=pl.cm.Blues, extent=[Xmin, Xmax, Ymin, Ymax])
    ax.plot(X, Y, 'k.', markersize=2)
    forceAspect(ax)

def kde1d(X, ax):
    """
    1D kernel density estimation plot.

    `X`
        An array of the variable
    `ax`
        matplotlib axes handler

    """
    Xmin = np.min(X)
    Xmax = np.max(X)
    x1 = np.linspace(Xmin, Xmax, 100)
    x2 = np.linspace(Xmin, Xmax, 50)
    kernel = stats.gaussian_kde(X)
    Z = kernel(x1)
    ax.hist(X, bins=x2, normed=True)
    ax.plot(x1, Z)
    #forceAspect(ax)

def pairwise_hist(chain):
    """
    Pairwise scatterplot (histogram)

    `chain`
        MCMC routine returned chain

    """
    n_sample, n_param = chain.shape
    fig_size = (3*n_param, 3*n_param)

    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    for i in range(n_param):
        for j in range(n_param):
            if i == j:
                # Plot the diagonal
                kde1d(chain[:,i], ax=axes[i,j])
            elif i < j:
                axes[i,j].axis('off')
            else:
                # Plot the samples as density map
                hist2d(chain[:,j], chain[:,i], ax=axes[i,j])
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
        axes[i,0].set_ylabel('parameter %d'%(i+1))
        axes[-1,i].set_xlabel('parameter %d'%(i+1))

    return fig

def pairwise_kde(chain):
    """
    Pairwise scatterplot (kernel density estimation)

    `chain`
        MCMC routine returned chain

    """
    n_sample, n_param = chain.shape
    fig_size = (3*n_param, 3*n_param)

    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    for i in range(n_param):
        for j in range(n_param):
            if i == j:
                # Plot the diagonal
                kde1d(chain[:,i], ax=axes[i,j])
            elif i < j:
                axes[i,j].axis('off')
            else:
                # Plot the samples as density map
                kde2d(chain[:,j], chain[:,i], ax=axes[i,j])
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
        axes[i,0].set_ylabel('parameter %d'%(i+1))
        axes[-1,i].set_xlabel('parameter %d'%(i+1))

    return fig


