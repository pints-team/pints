#
# Broyden-Fletcher-Goldfarb-Shanno algorithm
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from numpy.linalg import norm
from scipy.optimize.linesearch import line_search_wolfe2
#from scipy.optimize.linesearch import line_search_wolfe1
#from scipy.optimize.linesearch import line_search_BFGS
import pints


# TODO: move line search to abstract class
class BFGS_scipy(pints.LineSearchBasedOptimiser):
    """
    Broyden-Fletcher-Goldfarb-Shanno algorithm [1]

    [1] Liu, D. C.; Nocedal, J.
    On the Limited Memory BFGS Method for Large Scale Optimization.
    Mathematical Programming 1989, 45 (1),
    503-528. https://doi.org/10.1007/BF01589116.

    [2] Nocedal, J. Updating Quasi-Newton Matrices with Limited Storage.
    Math. Comp. 1980, 35 (151), 773-782.
    https://doi.org/10.1090/S0025-5718-1980-0572855-7.

    [3] Nash, S. G.; Nocedal, J. A Numerical Study of the Limited Memory
    BFGS Method and the Truncated-Newton Method for Large Scale Optimization.
    SIAM J. Optim. 1991, 1 (3), 358-372. https://doi.org/10.1137/0801023.

    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(BFGS_scipy, self).__init__(x0, sigma0, boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = self._x0
        self._fbest = float('inf')

        # approximate inverse hessian
        # initial the identity is used
        self._B = np.identity(self._n_parameters)

        # newton direction
        self._px = None

        # maximum number of correction matrices stored
        self._m = 5

        # Storage for vectors constructing the correction matrix
        # this is the advised way of storing them.
        self._S = np.zeros(shape=(self._n_parameters, self._m))
        self._Y = np.zeros(shape=(self._n_parameters, self._m))

        # number of accepted steps/ newton direction updates
        self.__k = 0

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = None
        self._current_dfdx = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        self._previous_f = None

        # parameters for wolfe conditions on line search

        # As c1 approaches 0 and c2 approaches 1, the line search
        # terminates more quickly.
        self._c1 = 1E-4  # Parameter for Armijo condition rule, 0 < c1 < 0.5
        self._c2 = 0.9  # Parameter for curvature condition rule, c1 < c2 < 1.0

        # boundary values of alpha
        self._minimum_alpha = 0.0
        self._maximum_alpha = float("inf")
        self._proposed_alpha = 0.001  # same default value as used in stan

        self.__current_alpha = 1.0

        self.__convergence = False

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        return self._fbest

    def wolfe_line_search_parameters(self):
        """
        Returns the  wolfe line search parameters this optimiser is using
        as a vector ``[c1, c2]``.
        As c1 approaches 0 and c2 approaches 1, the line search terminates
        more quickly. ``c1`` is the parameter for the Armijo condition
        rule, ``0 < c1 < 0.5``. ``c2 `` is the parameter for the
        curvature condition rule, ``c1 < c2 < 1.0``.
        """
        return (self._c1, self._c2)

    def max_correction_matrice_storage(self):
        """
        Returns ``m``, the maximum number of correction matrice for
        calculating the inverse hessian used and stored
        """
        return self._m

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Broyden-Fletcher-Goldfarb-Shanno (BFGS)'

    def needs_sensitivities(self):
        """ See :meth:`Optimiser.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`pints.TunableMethod.n_hyper_parameters()`. """
        return 2

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def set_hyper_parameters(self, x):
        """
        See :meth:`pints.TunableMethod.set_hyper_parameters()`.

        The hyper-parameter vector is ``[c1, c2, m]``.
        ``c1`` is the parameter for the Armijo condition rule,
        ``0 < c1 < 0.5``.
        ``c2`` is the parameter for the curvature condition rule,
        ``c1 < c2 < 1.0``.
        ``m`` is the number of maximum number of correction matrices
        that can be stored for the LM-BFGS update.
        """

        self.__set_wolfe_line_search_parameters(x[0], x[1])

    def __set_wolfe_line_search_parameters(self, c1: float, c2: float):
        """
        Sets the parameters for the wolfe conditions.

        Parameters
        ----------
        c1: float
        Parameter for the Armijo condition rule, ``0 < c1 < 0.5``.

        c2: float
        Parameter for the curvature condition rule, ``c1 < c2 < 1.0``.
        """
        if(0 < c1 < 0.5 and c1 < c2 < 1.0):
            self._c1 = c1
            self._c2 = c2
        else:
            cs = self.wolfe_line_search_parameters()
            print('Invalid wolfe line search parameters!!!')
            print('0 < c1 < 0.5 and c1 < c2 < 1.0')
            print('using default parameters: c1 = ', cs[0], ' c2 = ', cs[1])

    def __set_max_correction_matrice_storage(self, m: int):
        """
        Sets the parameters for the wolfe conditions.

        Parameters
        ----------
        m: int
        The maximum number of correction matrice for calculating the
        inverse hessian used and stored.
        """
        if(m == int(m)):
            self._m = m
            self._S = np.zeros(self._n_parameters, m)
            self._Y = np.zeros(self._n_parameters, m)
        else:
            print('Invalid value of m!!!\nm must be an integer')
            print('using default parameters: m = ', self._m)

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return self._xbest

    def __objective_function(self, point_alpha: float):
        '''
        For a given alpha this returns the values of the objective function.
        '''
        #point_alpha = self._current + alpha * self._px
        fs_alpha = self._evaluator.evaluate([point_alpha])
        f_alpha, dfdx_alpha = fs_alpha[0]

        return f_alpha

    def __gradients_function(self, point_alpha: float):
        '''
        For a given alpha this returns the values of the objective
        functions derivative.
        '''
        #point_alpha = self._current + alpha * self._px
        fs_alpha = self._evaluator.evaluate([point_alpha])
        f_alpha, dfdx_alpha = fs_alpha[0]

        # dfdalpha = np.matmul(np.transpose(dfdx_alpha), self._px)

        # print('dfdalpha: ', dfdalpha)
        # print('type dfdalpha: ', type(dfdalpha[0]))

        return dfdx_alpha

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """

        # print('')
        # print('in ask')
        # print('')

        if not self._running:
            self._proposed = np.asarray(self._x0)
        else:
            # line search using an algorithm from scipy to meet wolfe condtions
            results = line_search_wolfe2(f=self.__objective_function,
                                         myfprime=self.__gradients_function,
                                         xk=self._current, pk=self._px,
                                         gfk=self._current_dfdx,
                                         old_fval=self._current_f,
                                         old_old_fval=self._previous_f,
                                         c1=self._c1, c2=self._c2, maxiter=50)

            # line_search_BFGS below only checks the Armijo rule,
            # therefore it doesn't ensure the gradient is decreasing,
            # this can cause problems with the BFGS algorithm as
            # the Hessians from this approach may not be positive definite.

            # results = line_search_BFGS(f=self.__objective_function,
            #                            xk=self._current,
            #                            pk=self._px, gfk=self._current_dfdx,
            #                            old_fval=self._current_f, c1=self._c1,
            #                            alpha0=self.__current_alpha)

            # results = line_search_wolfe1(f=self.__objective_function,
            #                              fprime=self.__gradients_function,
            #                              xk=self._current, pk=self._px,
            #                              gfk=self._current_dfdx,
            #                              old_fval=self._current_f,
            #                              old_old_fval=self._previous_f,
            #                              c1=self._c1, c2=self._c2)

            self._proposed_alpha = results[0]
            # print('alpha: ', self._proposed_alpha)
            self._proposed = self._current + self._proposed_alpha * self._px

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True
        return [self._proposed]

    def tell(self, reply):
        """ See :meth:`Optimiser.tell()`. """

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Unpack reply
        proposed_f, proposed_dfdx = reply[0]
        proposed_f = proposed_f
        proposed_dfdx = np.asarray(proposed_dfdx)

        # We need the evaluation of the gradients before we can start the BFGS,
        # the first tell gets these.
        if self._current_f is None:

            # Move to proposed point
            self._current = self._proposed
            self._current_f = np.asarray(proposed_f)
            self._current_dfdx = np.asarray(proposed_dfdx)

            # Update newton direction
            # FIXME: is this right for inital newton direction???
            # if it isn't a desecnt direction the line searches will fail
            # i.e return alpha = none
            self._px = - np.matmul(self._B, self._current_dfdx)

        # If wolfe conditions meet the line search is stopped
        # and the hessian matrix and newton direction are updated by the
        # L-BFGS/BFGS approximation of the hessian described in reference [1]
        # [2], and [3]. If the line search has converged we also accept the
        # steps and update.
        else:

            # identity matrix
            I = np.identity(self._n_parameters)

            # We do this if we haven't exhausted existing memory yet, this is
            # identical to the BFGS algorithm
            if self.__k <= self._m - 1:
                k = self.__k
                # Defining the next column.
                self._S[:, k] = self._proposed - self._current
                self._Y[:, k] = proposed_dfdx - self._current_dfdx

                # Defining B_0. Scaling taken from [3].
                self._B = ((np.matmul(np.transpose(self._Y[:, k]),
                                      self._S[:, k])
                            / (norm(self._Y[:, k], ord=2) ** 2)) * I)

                # Updating inverse hessian.
                for k in range(self.__k + 1):

                    V = (I - np.matmul(self._Y[:, k],
                                       np.transpose(self._S[:, k]))
                         / np.matmul(np.transpose(self._Y[:, k]),
                                     self._S[:, k]))

                    self._B = np.matmul(np.transpose(V), np.matmul(self._B, V))
                    self._B += (np.matmul(self._S[:, k],
                                          np.transpose(self._S[:, k]))
                                / np.matmul(np.transpose(self._Y[:, k]),
                                            self._S[:, k]))

            # We have exhausted the limited memory and now enter
            # the LM-BFGS algorithm
            else:

                m = self._m - 1
                # Shifting everything one column to the left.
                self._S[:, 0:m] = self._S[:, 1:self._m]
                self._Y[:, 0:m] = self._Y[:, 1:self._m]

                # Renewing last column.
                self._S[:, m] = self._proposed - self._current
                self._Y[:, m] = proposed_dfdx - self._current_dfdx

                # Defining B_0. Scaling taken from [3].
                self._B = ((np.matmul(np.transpose(self._Y[:, m]),
                                      self._S[:, m])
                            / (norm(self._Y[:, m], ord=2) ** 2)) * I)

                # Updating inverse hessian.
                for k in range(self._m):

                    V = (I - np.matmul(self._Y[:, k],
                                       np.transpose(self._S[:, k]))
                         / np.matmul(np.transpose(self._Y[:, k]),
                                     self._S[:, k]))

                    self._B = np.matmul(np.transpose(V), np.matmul(self._B, V))
                    self._B += (np.matmul(self._S[:, k],
                                          np.transpose(self._S[:, k]))
                                / np.matmul(np.transpose(self._Y[:, k]),
                                            self._S[:, k]))

                self._B = ((np.matmul(np.transpose(self._Y[:, m]),
                                      self._S[:, m])
                            / (norm(self._Y[:, m], ord=2)**2)) * I)

            # Move to proposed point
            self._previous_f = self._current_f
            self._current = self._proposed
            self._current_f = np.asarray(proposed_f)
            self._current_dfdx = np.asarray(proposed_dfdx)

            # storing the accepted value of alpha
            self.__current_alpha = self._proposed_alpha

            # Update newton direction
            self._px = - np.matmul(self._B, self._current_dfdx)

            # incrementing the number of accepted steps
            self.__k += 1

        # Checking if all gradients ~ 0,
        # therefore the classical convergence test of a quasi-newton
        # or conjugate gradient method has been meet.
        if self.__convergence is not True:

            if norm(proposed_dfdx, ord=np.inf) <= 1e-6:

                self.__convergence = True
                print('')
                print(20 * '*' + ' Convergence after ',
                      self.__k, ' accepted steps!' + 20 * '*')
                print('||df/dx_i||inf <= 1e-6 with parameters:')
                print(self._proposed)
                print('error function evaluation: ', proposed_f)
                print('\nInverse Hessian matrix:\n', self._B)

        # Update xbest and fbest
        if self._fbest > proposed_f:
            self._fbest = proposed_f
            self._xbest = self._current

