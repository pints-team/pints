#
# ABC SMC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import random
from scipy.stats import multivariate_normal


class ABCAdaptivePMC(pints.ABCSampler):
    """
    TODO: description
    ABC-SMC Algorithm  See, for example, [1]_. In each iteration of the
    algorithm, the following steps occur::
        theta* ~ p_(t-1)(theta), i.e. sample parameters from previous
            intermediate distribution
        theta** ~ K(theta|theta*), i.e. perturb theta* to obtain to new point
        x ~ p(x|theta**), i.e. sample data from sampling distribution
        if s(x) < threshold_(t), theta* added to list of samples[t]

    After we have obtained nr_samples samples, t is advanced, and weights
    are calculated for samples[t-1]. At the last value for threshold,
    samples are returned whenever they are accepted.

    References
    ----------
    .. [1] "Sisson SA, Fan Y and Tanaka MM. Sequential Monte Carlo without
            likelihoods. Proc Natl Acad Sci USA, 104(6):1760-5, 2007."
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
                    # if cnt > 20:
                        # print("in loop, cnt="+str(cnt))
                        # print("theta_s"+str(theta_s)+", theta="+str(theta)+", var="+str(self._var))
                    cnt += 1
                if self._xs is None:
                    self._xs = [theta]
                else:
                    self._xs.append(theta)
        
        # print("finished ask with xs=" + str(self._xs))
        self._ready_for_tell = True
        return self._xs


    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        # print("started tell")
        if isinstance(fx, list):
            if self._t == 0:
                self._epsilon = self._calc_Q(fx)

                # Take only accepted values
                accepted = [a <= self._epsilon for a in fx]
                self._theta = [self._xs[c] for c, x in
                            enumerate(accepted) if x]
                self._fxs = [fx[c].tolist() for c, x in
                            enumerate(accepted) if x]
                # print("fxs init = " + str(self._fxs))
                self._weights = [1 / len(self._theta)] * len(self._theta)

                self._var = 2 * self._emp_var()
                self._t = self._t + 1
                # print("finished tell")
                return None
            else:
                self._n_weights = None
                print("until " + str(self._nr_samples - self._N_l))
                for i in range(self._nr_samples - self._N_l):
                    print("loop i="+str(i))
                    if self._n_weights is None:
                        self._n_weights = [self._compute_weights(i)]
                    else:
                        self._n_weights.append(self._compute_weights(i))
                    self._fxs.append(fx[i])
                print("finished weights loop")
                # print("thetas made of "+str(self._theta))
                # print("and "+str(self._xs))
                # print("weights made of "+str(self._weights))
                # print("and "+str(self._n_weights))
                # print("fxs len="+str(len(self._fxs)))
                s_accepted = [a <= self._epsilon for a in fx]
                p_acc = 1 / (self._nr_samples - self._N_l) * sum(s_accepted)
                # print("finished tell")
                # print("p_acc=" + str(p_acc) + ", t="+str(self._t))
                if p_acc <= self._p_acc_min:
                    self._theta.extend(self._xs)
                    return self._theta
                else:
                    # reduce xs and fx
                    self._epsilon = self._calc_Q(self._fxs)
                    print("epsilon="+str(self._epsilon))
                    o_accepted = [a <= self._epsilon for a in self._fxs]
                    # print("epsilon="+str(self._epsilon))
                    self._theta = [self._theta[c] for c, x in
                            enumerate(o_accepted) if x and c < self._N_l]
                    # print("acc theta="+str(self._theta))
                    accepted = [a <= self._epsilon for a in fx]
                    self._fxs = [self._fxs[c] for c, x in
                            enumerate(o_accepted) if x and c < self._N_l]
                    self._weights = [self._weights[c] for c, x in
                            enumerate(o_accepted) if x and c < self._N_l]
                    for c, x in enumerate(accepted):
                        if x:
                            self._theta.append(self._xs[c])
                            self._weights.append(self._n_weights[c])
                            self._fxs.append(fx[c])

                    
                    # print("resulting_errors"+str(self._fxs))
                    self._var = 2 * self._emp_var()
                    # print("remaining thetas="+str(self._theta))
                    # print("and their weights="+str(self._weights))
                    self._t = self._t + 1
                    return None

    def _compute_weights(self, i):
        w_sum = 0.0
        for j in range(self._N_l):
            w_sum += self._weights[j]

        norm_term = 0.0
        for j in range(self._N_l):
            norm_term += (self._weights[j] / w_sum) * \
                multivariate_normal(self._xs[i], self._var).pdf(self._theta[j])

        return np.exp(self._log_prior(self._xs[i])) / norm_term

    def _calc_Q(self, errors):
        err_c = errors.copy()
        err_c.sort()
        # print("errors="+str(err_c))
        # print("i="+str(self._N_l))
        i = self._N_l
        return err_c[i-1]

    def _gen_prev_theta(self):
        all_sum = 0.0

        for i in range(self._N_l):
            all_sum += self._weights[i]

        r = random.uniform(0, all_sum)
        
        i = 0
        sum = 0
        while i < self._N_l and sum <= r:
            sum += self._weights[i]
            i += 1

        return self._theta[i-1]

    def _emp_var(self):
        """ Computes the weighted empirical variance of self._theta. """
        ws = np.array(self._weights)
        ths = np.array(self._theta)
        # print("ths="+str(ths))
        # print("ws="+str(ws))
        # Compute weighted mean
        w_sum = sum(ws)

        w_mean = np.zeros(self._dim)
        for i in range(len(self._theta)):
            w_mean = w_mean + ws[i] * ths[i]
        
        w_mean /= w_sum

        # print("w_mean="+str(w_mean))
        
        # Compute the sum of the weights
        w_sum = 0.0
        for i in range(len(self._theta)):
            w_sum = w_sum + ws[i]

        # Compute sum of the squared weights
        w_sq_sum = 0.0
        for i in range(len(self._theta)):
            w_sq_sum = w_sq_sum + (ws[i] ** 2)

        # Compute the non-corrected variance estimation
        n_V = 0.0
        partial_mat = np.zeros((self._dim, self._dim))
        for i in range(len(self._theta)):
            diff = ths[i] - w_mean
            for j in range(self._dim):
                for k in range(self._dim):
                    partial_mat[j][k] = diff[j] * diff[k]
            n_V = n_V + ws[i] * partial_mat
        
        # Add correction term
        if w_sum ** 2 == w_sq_sum:
            print("1e-20 hit")
            e_var = (w_sum ** 2) / 1e-20 * n_V
        else:
            e_var = ((w_sum ** 2) / ((w_sum ** 2) - w_sq_sum)) * n_V
        
        # if(e_var[0][0] > 10):
            # print("e_var ="+str(e_var)+"weights="+str(ws)+", thetas="+str(ths))
        return e_var

    def set_N_l(self, N_l):
        """
        Setting N alpha.
        """
        self._N_l = N_l
    
    def set_p_acc_min(self, p_acc_min):
        self._p_acc_min = p_acc_min