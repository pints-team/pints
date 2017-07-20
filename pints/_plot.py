#
# 
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pylab as plt
import seaborn as sns
sns.set_style('white')

def plot_trace(param_samples, model, prior, fig_size=(6, 6)):
    names = prior.get_parameter_names()
    true_vals = model.get_params_from_vector(names)
    n_param = param_samples.shape[1]

    fig, axes = plt.subplots(n_param, 1, figsize=fig_size)
    for i in range(n_param):
        axes[i].plot(param_samples[:,i])
        axes[i].set_xticklabels([])
        if i < len(names):
            axes[i].set_ylabel(names[i])
        else:
            axes[i].set_ylabel(r'$\theta$')
    fig.tight_layout()
    return fig, axes


def scatter_diagonal(param_samples, model, prior, fig_size=(6, 6)):
    names = prior.get_parameter_names()
    true_vals = model.get_params_from_vector(names)

    n_param = param_samples.shape[1]
    fig, axes = plt.subplots(n_param, 1, figsize=fig_size)
    for i in range(n_param):
        sns.kdeplot(param_samples[:, i], ax=axes[i],shade=True)
        if i < len(names):
            axes[i].set_ylabel(names[i])
            axes[i].set_xlabel(names[i])
        else:
            axes[i].set_ylabel(r'$\theta$')
            axes[i].set_xlabel(r'$\theta$')

    fig.tight_layout(pad=0)
    return fig, axes


def scatter_grid(param_samples, model, prior, fig_size=(6, 6)):
    names = prior.get_parameter_names()
    true_vals = model.get_params_from_vector(names)

    n_param = param_samples.shape[1]
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    for i in range(n_param):
        for j in range(n_param):
            if i == j:
                sns.kdeplot(param_samples[:, i], ax=axes[i, j])
            elif i < j:
                axes[i, j].plot(param_samples[:, j], param_samples[:, i], '.', ms=2)
                if len(true_vals) > j:
                    axes[i, j].plot([true_vals[j]], [true_vals[i]], 'r*')
            else:
                sns.kdeplot(param_samples[:, j], param_samples[:, i], cmap='Blues',
                            shade=True, shade_lowest=False, n_levels=10, ax=axes[i, j])
                if len(true_vals) > j and len(true_vals) > i:
                    axes[i, j].plot([true_vals[j]], [true_vals[i]], 'r*')
            if i < n_param-1:
                axes[i, j].set_xticklabels([])
            if j > 0:
                axes[i, j].set_yticklabels([])
        if i < len(names):
            axes[i, 0].set_ylabel(names[i])
            axes[-1, i].set_xlabel(names[i])
        else:
            axes[i, 0].set_ylabel(r'$\theta$')
            axes[-1, i].set_xlabel(r'$\theta$')

    fig.tight_layout(pad=0)
    return fig, axes




