#
# ABC Adaptive PMC
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy.stats import multivariate_normal

class ABCAdaptivePMC(pints.ABCSampler):
    """
    ABC Adaptive PMC Algorithm  See, for example, [1]_. First iteration 
    samples ``n_samples`` from the prior, simulates data for each sample,
    and computes and stores the distance function value :math:`\rho_i` from
    the original data. For the next generation, only samples with the
    lowest ``N_l`` distance values. For future generations the following
    procedure is followed for samples ``N_l`` until ``N``:

    .. math::
        \begin{align}
        & \theta^* \sim p_{t-1}(\theta) \textrm{, i.e. sample parameters from
        previous intermediate distribution} \\
        & \theta_i \sim K(\theta|\theta^{*}), \textrm{i.e. perturb }
        \theta^{*} \textrm{ to     obtain to new point } x \sim
        p(x|\theta^{**})\textrm{, i.e. sample data from sampling
        distribution} \\
        & \rho_i = \rho(S(x), S(y)) \textrm{ where $y$ is the original data}
        \end{align}
    
    After all :math:`N - N_l` samples are calculated and their distances,
    we can compute the acceptance rate:
    
    ..math::
        \begin{equation}
            p_{acc} = \frac{1}{N - N_{\alpha}} \sum_{k=N_{\alpha} + 1}^N
            \mathbb{1}(\rho_i^{t-1} < \epsilon_{t-1})        
        \end{equation}

    When the user input ``p_acc_min`` is greater than :math:`p_{acc}`
    we return th generation. Otherwise, we keep the samples with
    ``N_l`` smallest :math:`\rho_i` and apply the same procedure
    until we find a small enough acceptance rate.

    References
    ----------
    .. [1] Lenormand, Maxime, Franck Jabot, and Guillaume Deffuant. Adaptive
           approximate Bayesian computation for complex models. Computational
           Statistics 28.6 (2013): 2777-2796.
           https://doi.org/10.1007/s00180-013-0428-3
    """

    def __init__(self, log_prior, perturbation_kernel=None):

        self._log_prior = log_prior
        self._samples = [[]]
        self._accepted_count = 0
        self._weights = []
        self._threshold = 1
        self._e_schedule = [1]
        self._nr_samples = 100
        self._xs = None
        self._ready_for_tell = False
        self._t = 0
        self._p_acc_min = 0.5
        self._dim = log_prior.n_parameters()
        self._cnt = 0

        dim = log_prior.n_parameters()

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'ABC-Adaptive-PMC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('ask called before tell.')
        if self._t == 0:
            self._xs = self._log_prior.sample(n_samples).tolist()
            self._nr_samples = len(self._xs)
        else:
            self._xs = None
            for i in range(self._N_l + 1, n_samples + 1):
                done = False
                cnt = 0
                while not done:
                    theta_s = self._gen_prev_theta()
                    theta = np.random.multivariate_normal(theta_s, self._var)
                    done = self._log_prior(theta) != np.NINF
                    cnt += 1
                    if not done:
                        self._cnt += 1
                if self._xs is None:
                    self._xs = [theta]
                else:
                    self._xs.append(theta)
        
        print("ask cnt = " + str(self._cnt))
        self._ready_for_tell = True
        return self._xs


    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        if isinstance(fx, list):
            if self._t == 0:
                self._epsilon = self._calc_Q(fx)

                # Take only accepted values
                accepted = [a <= self._epsilon for a in fx]
                self._theta = [self._xs[c] for c, x in
                            enumerate(accepted) if x]
                self._fxs = [fx[c].tolist() for c, x in
                            enumerate(accepted) if x]
                self._weights = [1 / len(self._theta)] * len(self._theta)

                self._var = 2 * self._emp_var()
                self._t = self._t + 1
                return None
            else:
                self._n_weights = None
                s_L = len(self._fxs)
                for i in range(self._nr_samples - self._N_l):
                    if self._n_weights is None:
                        self._n_weights = [self._compute_weights(i)]
                    else:
                        self._n_weights.append(self._compute_weights(i))
                    self._fxs.append(fx[i])
                s_accepted = [a <= self._epsilon for a in fx]
                p_acc = 1 / (self._nr_samples - self._N_l) * sum(s_accepted)
                if p_acc <= self._p_acc_min:
                    self._theta.extend(self._xs)
                    return self._theta
                else:
                    # reduce xs and fx
                    self._epsilon = self._calc_Q(self._fxs)
                    print("epsilon="+str(self._epsilon))
                    o_accepted = [a <= self._epsilon for a in self._fxs]
                    # In case there are multiple values with the error
                    # equal to the error threshold
                    acc_l = np.sum(o_accepted)
                    if acc_l > self._N_l:
                        dif = acc_l - self._N_l

                        for i in range(len(o_accepted)):
                            if self._fxs[i] == self._epsilon:
                                if dif > 0:
                                    o_accepted[i] = False
                                    dif = dif - 1
                    self._theta = [self._theta[c] for c, x in
                            enumerate(o_accepted) if x and c < s_L]
                    self._fxs = [self._fxs[c] for c, x in
                            enumerate(o_accepted) if x and c < s_L]
                    self._weights = [self._weights[c] for c, x in
                            enumerate(o_accepted) if x and c < s_L]
                    for c, x in enumerate(o_accepted):
                        if c >= s_L and x:
                            self._theta.append(self._xs[c - s_L])
                            self._weights.append(self._n_weights[c - s_L])
                            self._fxs.append(fx[c - s_L])
                    
                    self._var = 2 * self._emp_var()
                    self._t = self._t + 1
                    return None

    def _compute_weights(self, i):
        w_sum = 0.0
        for j in range(self._N_l):
            w_sum += self._weights[j]

        norm_term = 0.0
        for j in range(self._N_l):
            norm_term += (self._weights[j] / w_sum) * \
                multivariate_normal(self._xs[i], self._var, allow_singular=True).pdf(self._theta[j])

        return np.exp(self._log_prior(self._xs[i])) / norm_term

    def _calc_Q(self, errors):
        err_c = errors.copy()
        err_c.sort()
        i = self._N_l
        return err_c[i-1]

    def _gen_prev_theta(self):
        all_sum = 0.0
        for i in range(len(self._weights)):
            all_sum += self._weights[i]

        r = np.random.uniform(0, all_sum)
        
        i = 0
        sum = 0
        while i < len(self._weights) and sum <= r:
            sum += self._weights[i]
            i += 1

        return self._theta[i-1]

    def _emp_var(self):
        """ Computes the weighted empirical variance of self._theta. """
        ws = np.array(self._weights)
        ths = np.array(self._theta)
        # Compute weighted mean
        w_sum = sum(ws)

        for i in range(len(self._theta)):
            ws[i] = ws[i] / w_sum
        
        w_sum = 1

        w_mean = np.zeros(self._dim)
        for i in range(len(self._theta)):
            w_mean = w_mean + ws[i] * ths[i]
        
        w_mean /= w_sum

        print("w_mean="+str(w_mean))
        
        # Compute sum of the squared weights
        w_sq_sum = 0.0
        for i in range(len(self._theta)):
            w_sq_sum = w_sq_sum + (ws[i] ** 2)

        # Compute the non-corrected variance estimation
        n_V = None
        for i in range(len(self._theta)):
            diff = np.array([ths[i] - w_mean])
            partial_mat = diff * np.transpose(diff)
            if n_V is None:
                n_V = ws[i] * partial_mat
            else:
                n_V = n_V + ws[i] * partial_mat
        
        # Add correction term
        if w_sum ** 2 == w_sq_sum:
            e_var = (w_sum) / 1e-20 * n_V
        else:
            e_var = (w_sum/ ((w_sum ** 2) - w_sq_sum)) * n_V
        
        # if(e_var[0][0] > 10):
            # print("e_var ="+str(e_var)+"weights="+str(ws)+", thetas="+str(ths))
        print("resulting var="+str(2 * e_var))
        return e_var

    def set_N_l(self, N_l):
        """
        Setting N alpha.
        """
        self._N_l = N_l
    
    def set_p_acc_min(self, p_acc_min):
        self._p_acc_min = p_acc_min