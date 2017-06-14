import pylab as plt
import seaborn as sns
sns.set_style('white')

def scatter_grid(param_samples, true_vals, fig_size=(6, 6)):
    n_param = param_samples.shape[1]
    print 'scatter_grid: n_param = ',n_param
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
            #axes[i, j].set_xticklabels([])
            #axes[i, j].set_yticklabels([])
        axes[i, 0].set_ylabel('$u_{0}$'.format(i), fontsize=14)
        axes[-1, i].set_xlabel('$u_{0}$'.format(i), fontsize=14)
    fig.tight_layout(pad=0)
    return fig, axes




