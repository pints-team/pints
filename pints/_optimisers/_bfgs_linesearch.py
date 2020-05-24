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
import pints


class BFGS(pints.LineSearchBasedOptimiser):
    """
    Broyden-Fletcher-Goldfarb-Shanno algorithm [2], [3], [4]

    The Hager-Zhang line search algorithm [1] is implemented in this class
    # TODO: when this is working move everything to an abstract class

    [1] Hager, W. W.; Zhang, H. Algorithm 851: CG_DESCENT,
    a Conjugate Gradient Method with Guaranteed Descent.
    ACM Trans. Math. Softw. 2006, 32 (1), 113-137.
    https://doi.org/10.1145/1132973.1132979.

    [2] Liu, D. C.; Nocedal, J.
    On the Limited Memory BFGS Method for Large Scale Optimization.
    Mathematical Programming 1989, 45 (1),
    503-528. https://doi.org/10.1007/BF01589116.

    [3] Nocedal, J. Updating Quasi-Newton Matrices with Limited Storage.
    Math. Comp. 1980, 35 (151), 773-782.
    https://doi.org/10.1090/S0025-5718-1980-0572855-7.

    [4] Nash, S. G.; Nocedal, J. A Numerical Study of the Limited Memory
    BFGS Method and the Truncated-Newton Method for Large Scale Optimization.
    SIAM J. Optim. 1991, 1 (3), 358-372. https://doi.org/10.1137/0801023.

    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(BFGS, self).__init__(x0, sigma0, boundaries)

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = self._x0
        self._fbest = float('inf')

        # Number of iterations run
        self._iterations = 0

        # Parameters for wolfe conditions on line search

        # As c1 approaches 0 and c2 approaches 1, the line search
        # terminates more quickly.
        self._c1 = 1E-4  # Parameter for Armijo condition rule, 0 < c1 < 0.5
        self._c2 = 0.9  # Parameter for curvature condition rule, c1 < c2 < 1.0

        # boundary values of alpha
        self._minimum_alpha = 0.0
        self._maximum_alpha = float("inf")
        self._proposed_alpha = 0.001  # same default value as used in stan

        self.__first_update_step_not_completed = True
        self.__update_step_not_completed = True
        self.__performing_line_search = False

        # Increase allowed between accepted positions when using approximate
        # wolfe conditions, this takes into acccount machine error and
        # insures decreasing.
        self.epsilon = 1E-6

        # range (0, 1), used in the ``self.__update()`` and
        # ``self.__initial_bracket()`` when the potential intervals violate
        # the opposite slope condition (see function definition)
        self.theta = 0.5

        self.__gamma = 0.66

        # range (0, 1) small factor used in initial guess of step size
        self.__ps_0 = 0.01
        # range (0, 1) small factor used in subsequent guesses of step size
        self.__ps_1 = 0.1
        # range (1, inf) factor used in subsequent guesses of step size
        self.__ps_2 = 2.0

        self.__current_alpha = None

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

        # number of steps in the current line search iteration
        self.__j = 0
        self.__logic_steps_left = 0

        # maximum number of line search steps before a successful point

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = None
        self._current_dfdx = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        self.__convergence = False

        # logic for passing to tell at the right moments
        self.__1st_wolfe_check_needed = False
        self.__1st_wolfe_check_done = False
        self.__2nd_wolfe_check_needed = False
        self.__2nd_wolfe_check_done = False
        self.__need_update = False
        self.__converged_ls = False

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

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """

        # print('')
        # print('in ask')
        # print('')

        if not self._running:
            self._proposed = np.asarray(self._x0)
        else:

            if self.__j == 0:
                # working out an initial stepsize value alpha
                alpha_0 = self.__initialising(k=self.__k,
                                              alpha_k0=self.__current_alpha)
                print('alpha_initial: ', alpha_0)
                # Creating an initial bracketing interval of alpha values
                # satisfying the opposite slope condition (see function
                # docstring) beginning with initial guess [0, alpha_initial].
                bracket = (self.__initial_bracket(c=alpha_0))
                self._minimum_alpha = bracket[0]
                self._maximum_alpha = bracket[1]
                self._updated_minimum_alpha = self._minimum_alpha
                self._updated_maximum_alpha = self._maximum_alpha

            # Looping while wolfe conditions don't need to be checked
            # and the line search hasn't converged.
            while (~(self.__1st_wolfe_check_needed) &
                   ~(self.__2nd_wolfe_check_needed) & ~(self.__converged_ls)):

                # secant squared step of the line search

                # ***************************************************************
                # a, b = self.__secant2(self._minimum_alpha,
                #                       self._maximum_alpha)
                a = self._updated_minimum_alpha
                b = self._updated_maximum_alpha
                c = self._proposed_alpha

                # checking if the bracketing interval has converged
                self.__converged_ls = self.__very_close(a, b)
                self.__logic_steps_left = 'not started'
                if self.__converged_ls:
                    self.__logic_steps_left = ' converged in ls '
                # if converged is True don't do anything more.

                # ************ beginning of secant squared see [1] ************

                # Preforming a secant to propose a value of alpha.
                if (~(self.__1st_wolfe_check_done) &
                   ~(self.__2nd_wolfe_check_done) & ~(self.__converged_ls)):

                    # step S1 in [1]
                    self._proposed_alpha = self.__secant_for_alpha(a, b)
                    # passing to tell to check wolfe conditions
                    self.__1st_wolfe_check_needed = True

                    # checking the proposed point is in range
                    if self._proposed_alpha < a or self._proposed_alpha > b:
                        # If the point is out of range there is no need to
                        # check wolfe conditions.
                        self.__1st_wolfe_check_needed = False
                        self.__1st_wolfe_check_done = True

                    self.__logic_steps_left = 2
                    self.__j += 1 / 3

                elif (self.__1st_wolfe_check_done &
                      ~(self.__2nd_wolfe_check_done) & ~(self.__converged_ls)):

                    # (typically) updating one side of the
                    # bracketing interval
                    A, B = self.__update(a, b, c)
                    # end of step S1 in [1]

                    # (typically) updating the otherside side of
                    # the bracketing interval
                    if c == B:
                        # S2 in [1]
                        # Preforming a secant to propose a value of alpha.
                        self._proposed_alpha = self.__secant_for_alpha(b, B)

                        # checking the proposed point is in range
                        if self._proposed_alpha < A | self._proposed_alpha > B:
                            # If the point is out of range there is no need to
                            # check wolfe conditions.
                            self.__2nd_wolfe_check_needed = False
                            self.__2nd_wolfe_check_done = True
                        else:
                            self.__2nd_wolfe_check_needed = True
                            self.__need_update = True
                    elif c == A:
                        # S3 in [1]
                        # Preforming a secant to propose a value of alpha.
                        self._proposed_alpha = self.__secant_for_alpha(a, A)

                        # checking the proposed point is in range
                        if self._proposed_alpha < A | self._proposed_alpha > B:
                            # If the point is out of range there is no need to
                            # check wolfe conditions.
                            self.__2nd_wolfe_check_needed = False
                            self.__2nd_wolfe_check_done = True
                        else:
                            self.__2nd_wolfe_check_needed = True
                            self.__need_update = True
                    else:
                        # No new point has been proposed therefore there
                        # is no need to check the wolfe conditions.
                        self.__2nd_wolfe_check_needed = False
                        self.__2nd_wolfe_check_done = True

                    self._updated_minimum_alpha = A
                    self._updated_maximum_alpha = B

                    self.__logic_steps_left = 1
                    self.__j += 1 / 3

                elif (self.__1st_wolfe_check_done &
                      self.__2nd_wolfe_check_done & ~(self.__converged_ls)):

                    # S4 in [1], this is only preformed if S2 or S3 was done
                    # and the propsed point was in range.
                    if self.__need_update:
                        a, b = self.__update(a, b, c)
                        self.__need_update = False

                    # ***************** end of secant squared *****************

                    # preforming steps L2 from [1]
                    # determing whether a bisection step should be preformed
                    # i.e if self.__secant_for_interval() didn't shrink the
                    # bracketing interval by the propotion self.__gamma
                    new_width = b - a
                    old_width = (self._maximum_alpha - self._minimum_alpha)
                    # print('lower_alpha: ', self._updated_minimum_alpha,
                    # 'updated_upper_alpha: ', self._updated_maximum_alpha)
                    # print('_maximum_alpha: ', self._minimum_alpha,
                    # ' maximum alpha: ', self._maximum_alpha)
                    if new_width > self.__gamma * old_width:
                        # preforming bisection
                        #print('preforming bisection')
                        c = (a + b) / 2.0
                        a, b = self.__update(a=a, b=b, c=c)
                        #print('bisected_lower: ',a,' bisected upper: ', b)
                        #print('finished bisection')

                    # preforming steps L3 from [1]
                    # updating bracketing interval
                    self._minimum_alpha, self._maximum_alpha = a, b
                    self.__j += 1 / 3
                    self.__logic_steps_left = 0

                    # reset logic
                    self.__1st_wolfe_check_needed = False
                    self.__1st_wolfe_check_done = False
                    self.__2nd_wolfe_check_needed = False
                    self.__2nd_wolfe_check_done = False

                # print('line step loops: ', self.__j,
                #       ' logic steps left: ', self.__logic_steps_left)
                # print('lower_alpha: ', self._updated_minimum_alpha,
                #       'updated_upper_alpha: ', self._updated_maximum_alpha)
                # print('_maximum_alpha: ', self._minimum_alpha,
                #       ' maximum alpha: ', self._maximum_alpha)
                # print('converged: ',
                #       self.__very_close(self._updated_minimum_alpha,
                #                         self._updated_maximum_alpha))
                # *********************** CONTINUE LOOP ***********************

            if self.__converged_ls:
                self._proposed = (self._current +
                                  self._updated_maximum_alpha * self._px)
            else:
                self._proposed = (self._current +
                                  self._proposed_alpha * self._px)

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True

        # print('')
        # print('finished ask')
        # print('')
        # Return proposed points (just the one) in the search space to evaluate
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

            # resetting the number of steps in the current
            # line search iteration.
            self.__j = 0

        # Checking if exact wolfe conditions.
        proposed_grad = np.matmul(np.transpose(proposed_dfdx), self._px)

        wolfe_curvature = (proposed_grad >=
                           self._c2 *
                           np.matmul(np.transpose(self._current_dfdx),
                                     self._px))

        exact_wolfe_suff_dec = (self._c1 *
                                np.matmul(np.transpose(self._current_dfdx),
                                          self._px)
                                >= proposed_f - self._current_f)

        exact_wolfe = (exact_wolfe_suff_dec and wolfe_curvature)

        # Checking if approximate wolfe conditions are meet.
        approx_wolfe_suff_dec = ((2.0 * self._c1 - 1.0) *
                                 np.matmul(np.transpose(self._current_dfdx),
                                           self._px) >= proposed_grad)

        approx_wolfe_applies = proposed_f <= self._current_f + self.epsilon

        approximate_wolfe = (approx_wolfe_suff_dec and wolfe_curvature
                             and approx_wolfe_applies)

        # If wolfe conditions meet the line search is stopped
        # and the hessian matrix and newton direction are updated by the
        # L-BFGS/BFGS approximation of the hessian described in reference [2]
        # [3], and [4]. If the line search has converged we also accept the
        # steps and update.
        if exact_wolfe or approximate_wolfe or self.__converged_ls:

            print('Number of accepted steps: ', self.__k)
            print('step sized alpha: ', self._proposed_alpha, ' accepted')
            print('updating Hessian and changing newton direction')

            # Updating inverse hessian.

            # identity matrix
            I = np.identity(self._n_parameters)

            # We do this if we haven't exhausted existing memory yet, this is
            # identical to the BFGS algorithm
            if self.__k <= self._m - 1:
                k = self.__k
                # Defining the next column.
                self._S[:, k] = self._proposed - self._current
                self._Y[:, k] = proposed_dfdx - self._current_dfdx

                # Defining B_0. Scaling taken from [4].
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

                # Defining B_0. Scaling taken from [4].
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
            self._current = self._proposed
            self._current_f = np.asarray(proposed_f)
            self._current_dfdx = np.asarray(proposed_dfdx)

            # storing the accepted value of alpha
            self.__current_alpha = self._proposed_alpha

            # if self.__k == 0:
            #     print('\nThe first accepted value of alpha was: ',
            #           self.__current_alpha)
            #     print('set initial alpha to this in subsequent runs' +
            #           'to speed up computation')

            # Update newton direction
            self._px = - np.matmul(self._B, self._current_dfdx)

            # incrementing the number of accepted steps
            self.__k += 1

            # Resetting the number of steps in the current line search
            # iteration as we have accepted a value and completed this
            # line search.
            self.__j = 0

            # resetting line search logic
            self.__1st_wolfe_check_needed = False
            self.__1st_wolfe_check_done = False
            self.__2nd_wolfe_check_needed = False
            self.__2nd_wolfe_check_done = False
            self.__need_update = False

        else:
            # wolfe conditions haven't been meet so we continue the line search
            if self.__1st_wolfe_check_needed:
                self.__1st_wolfe_check_needed = False
                self.__1st_wolfe_check_done = True
            if self.__2nd_wolfe_check_needed:
                self.__2nd_wolfe_check_needed = False
                self.__2nd_wolfe_check_done = True

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

        print('')
        print('Hessian updates: ', self.__k, ' line steps: ', self.__j,
              ' propose alpha: ', self._proposed_alpha, ' propose points: ',
              self._proposed)

    def __update(self, a, b, c):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Updates the bracketing boundary values of alpha. Ensuring the opposite
        slope conditions are obeyed.

        The opposite slope conditions are:
                                φ(a) ≤ φ(0) + epsilon,
                                φ'(a) < 0 (w.r.t every parameter),
                                φ'(b) ≥ 0 (w.r.t every parameter).
        where φ(α) = f(x+α*px), f() is the function binging minimised, x is
        the current mparameter set, px is the newton direction, and alpha is
        the step size.

        In the first condition, epsilon is a small positive constant. The
        condition demands that the function at the left end point be not much
        bigger than the starting point (i.e. alpha = 0). This is an easy to
        satisfy condition because by assumption, we are in a direction where
        the function value is decreasing. The second and third conditions
        together demand that there is at least one zero of the derivative in
        between a and b. In addition to the interval, the update algorithm
        requires a third point to be supplied. Usually, this point would lie
        within the interval [a, b]. If the point is outside this interval,
        the current interval is returned. If the point lies within the
        interval, the behaviour of the function and derivative value at this
        point is used to squeeze the original interval in a manner that
        preserves the opposite slope conditions.

        :param a: lower bound of alpha
        :param b: upper bound of alpha
        :param c: proposed value of alpha
        :param dfdx_c: vector of gradients at the proposed point i.e alpha = c
        :param f_c: function evaluation at the proposed point i.e alpha = c
        :param f_0: function evaluation at the previous point i.e alpha = 0
        '''

        # Check c is within the bracket conditions (steps U0 from [1]).
        if c < a or c > b:
            # if the proposed point is not in range we return
            # the old brackets unmodified
            return a, b
        else:

            # evaluate function for alpha = c i.e at the proposed boundary
            # point_c = self._current + c * self._px
            # fs_c = self._evaluator.evaluate([point_c])
            f_c, dfdx_c = self.obj_and_grad_func(c)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point
            # (steps U1 from [1]).
            if dfdx_c >= 0.0:
                # Updating the upper bound.
                return a, c

            # Checking if the opposite slope condition at the lower bound is
            # obeyed by the proposed point, if so it is a valid lower bound
            # (steps U2 from [1]).
            elif dfdx_c < 0.0 and f_c <= self._current_f + self.epsilon:
                # Updating the lower bound.
                return c, b

            # The proposed point doesn't obey the opposite slope condition
            # i.e. dfdx_c < 0.0 and f_c > self._current_f + self.epsilon.
            # Checking this is unnecessary as it is the opposite to the above
            # conditions. A secant/bisect can narrow down an interval between
            # the current lower bound and the trial point c.
            else:
                b = c
                return self.__bisect_or_secant(a, b)

    def __bisect_or_secant(self, a: float, b: float):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Actual implementation of secant (or bisetc if `self.theta` = 0.5)
        given a bracketing intervale [a, b] used in `__update()` and
        `__initial_bracketing_interval()`. (steps U3a-c from [1])


        :param a: lower bound of alpha
        :param b: upper bound of alpha
        :param c: proposed value of alpha
        '''

        secant = True

        # FIXME:
        # problem is in this while loop a and b merge but don't satisfy
        # opposite slope rule for upper,
        # probably should return when difference between a and b almost
        # nothing???

        # The interval needs updating if the upper bracket has a negative
        # slope and the value of the function at that point is too high.
        # It is not a valid lower bracket but along with the current
        # lower bracket, it encloses another minima. The below function
        # is a loop which tries to narrow the interval so that it
        # satisfies the opposite slope conditions.

        # (steps U3 from [1])

        while secant is True:
            # according to [1]] this while loop is guaranteed to terminate
            # as the intervale between a and b tends to zero

            # Preforming secant (if theta = 0.5 as is default this is a
            # bisection) to propose a new point which hopeful obeys the
            # opposite slope conditions.
            # (steps U3a from [1])
            d = (1.0 - self.theta) * a + self.theta * b

            # Evaluating function for alpha = d i.e at the proposed boundary.
            # point_d = self._current + d * self._px
            # fs = self._evaluator.evaluate([point_d])
            # f_d, dfdx_d = fs[0]

            f_d, dfdd = self.obj_and_grad_func(d)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point.
            # If the proposed point has a positive slope, then we have found a
            # suitable upper bound to bracket a minima within opposite slopes.
            # (still steps U3a from [1])
            converged = self.__very_close(d, b) or self.__very_close(d, a)
            if dfdd >= 0.0 or converged:
                secant = False
                # Updating the upper bound.
                return a, d

            # Checking if the opposite slope condition at the lower bound is
            # obeyed by the proposed point.
            # If the proposed point has a negative slope and the function value
            # at that point is small enough, we can use it as a new lower bound
            # to narrow down the interval.
            # (steps U3b from [1])
            elif dfdd < 0.0 and f_d <= self._current_f + self.epsilon:
                # Updating the lower bound.
                a = d

            # The proposed point doesn't obey the opposite slope condition
            # i.e. dfdx_c < 0.0 and f_c > self._current_f + self.epsilon
            # Checking this is unnecessary as it is the opposite to the above
            # conditions. We are therefore in the same situation as when we
            # started the loop so we update the upper bracket and continue.
            # (steps U3c from [1])
            else:
                b = d

    def __secant_for_alpha(self, a, b):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Preforms a secant step to propose a value of alpha. This is the same as
        the secant routine described in [1].

        :param a: lower bound of alpha
        :param b: upper bound of alpha
        '''

        # Evaluating function for alpha = a to obtain gradients.
        f_a, dfda = self.obj_and_grad_func(a)

        # Evaluating function for alpha = b to obtain gradients.
        f_b, dfdb = self.obj_and_grad_func(b)

        # Preforming secant.
        numerator = a * dfdb - b * dfda
        denominator = dfdb - dfda

        return float(numerator / denominator)

    def __initial_bracket(self, c, rho=5):
        '''
        This function is part of the Hager-Zhang line search method [1].

        This function is used to generate an initial interval [a, b] for alpha
        satisfying the opposite slope conditions and therefore bracketing
        the minimum, beginning with the initial guess [0, c].
        The opposite slope conditions:
                                φ(a) ≤ φ(0) + epsilon,
                                φ'(a) < 0 (w.r.t every parameter),
                                φ'(b) ≥ 0 (w.r.t every parameter).
        where φ(α) = f(x+α*px), f() is the function binging minimised, x is
        the current parameter set, px is the newton direction, and alpha is
        the step size.

        This is the same as the bracket routine described in [1] as steps B0-3

        :param c: initial guess for maximum value of alpha
        :param row: range (1, ∞), expansion factor used in the bracket rule to
                    increase the upper limit c (c_j+1 = row*c_j) until a
                    suitable interval is found that contains the minimum.
        '''

        # (steps B0 from [1])
        # Initiating a list of proposed boundary values of alpha.
        c = [c]
        # Initiating a list of error function evaluations at
        # the proposed boundary values of alpha.
        f_c = []
        # Initiating lower bound.
        a = 0

        # Initiating an iteration counter for the below while loop.
        j = 0
        bracketing = True

        while bracketing:

            # Evaluating function for alpha = c[j]
            # i.e at the proposed boundary.
            f_cj, dfdc_j = self.obj_and_grad_func(c[j])

            # Appending the error function evaluations at proposed boundary
            # values of alpha.
            f_c.append(f_cj)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point. If the slope at the propsed point
            # is positive, then the given points already bracket a minimum.
            # (steps B1 from [1])
            if dfdc_j >= 0.0:
                # Setting the upper bound.
                b = c[j]

                bracketing = False

                # Checking if the non derivative opposite slope condition at
                # the lower bound is obeyed by any of the previously evaluated
                # points and returning the value for which the boundary
                # conditions are as close together as possible.
                for i in range(1, j + 1):
                    if f_c[j - i] <= self._current_f + self.epsilon:
                        a = c[j - i]
                        return a, b
                return a, b

            # Checking if the proposed point doesn't obey the opposite slope
            # condition. This means the upper bracket limit almost works as a
            # new lower limit but the objective function(f_cj) is too large.
            # We therefore need to preform a secant/bisect but the minimum is
            # not yet bracketed.
            # (steps B2 from [1])
            elif dfdc_j < 0.0 and f_cj > self._current_f + self.epsilon:
                # The interval needs updating if the upper bracket has a
                # negative slope and the value of the function at that point
                # is too high. It is not a valid lower bracket but along with
                # the current lower bracket, it encloses another minima. The
                # below function tries to narrow the interval so that it
                # satisfies the opposite slope conditions.
                bracketing = False
                return self.__bisect_or_secant(0.0, c[j])

            # The proposed point obeys the opposite slope condition
            # at the lower bound
            # i.e. dfdx_d < 0.0 and f_d <= self._current_f + self.epsilon.
            # Checking this is unnecessary as it is the opposite to
            # the above conditions. This means the bracket interval needs
            # expanding to ensure a minimum is bracketed.
            # (steps B3 from [1])
            else:
                # Increasing the proposed point by a factor of row to attempt
                # to bracket minimum and trying again.
                c.append(rho * c[j])

                # Increamenting the iteration counter.
                j += 1

    def __initialising(self, k, alpha_k0):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Generate the starting guess of alpha, 'c', used by
        ``__initial_bracket()``. This is the same as the routine
        called initial and described as I0-2 in [1].

        :param k: number of accepted steps/newton direction updates that
        have taken place.
        :param alpha_k0: the alpha value used by the previously accepted
        steps/newton direction
        update. If k = 0 this is the initial alpha the user wants to be used
        '''
        QuadStep = False

        # Small factor used in initial guess of step size, range (0, 1).
        # As the initial guess is very crude this is a very small factor
        # to keep the initial step short and close the starting parameters
        # self.x_0.
        ps_0 = self.__ps_0

        # Small factor used in subsequent guesses of step size
        # if Quadstep is true, range (0, 1).
        # TODO: this hasn't been implement yet
        ps_1 = self.__ps_1

        # Sacling factor used in subsequent guesses of step size,
        # range (1, inf).
        ps_2 = self.__ps_2

        # For the first line search do the following
        # (step I0 in [1])
        if k == 0:

            if alpha_k0 is not None:
                # returning user specified initial alpha
                return alpha_k0

            # (step I0a in [1])
            elif np.all(self._x0 != 0.0):
                # Crude step size estimate
                # :math: \alpha = ps_0*||x_0||_\inf / ||dfdx||_\inf
                return ((ps_0 * norm(self._x0, ord=np.inf))
                        / (norm(self._current_dfdx, ord=np.inf)))

            # If self._x0 = 0.0 the above statement would give alpha = 0,
            # hence we use the following estimation.
            # (step I0b in [1])
            elif self._current_f != 0:
                # Crude step size estimate
                # :math: \alpha = ps_0*|f(x_0)| / ||dfdx||^2
                return ((ps_0 * abs(self._current_f))
                        / (pow(norm(self._current_dfdx, ord=2), 2.0)))

            # Otherwise self._current_f = 0.0 and we are already at the
            # minimum therefore return alpha = 1.0 it should not matter what we
            # return in this case as if self._current_f = 0.0 the gradients
            # will also equal zero.
            # (step I0c in [1])
            else:
                return 1.0

        # TODO: implement the below option using a quadratic interpolant ???
        # everything will work without this
        # (step I1 in [1])
        elif QuadStep:

            # point_current_scaled = self._current + ps_1*alpha_k0* self._px
            # fs = self._evaluator.evaluate([point_current_scaled])
            f, df = self.obj_and_grad_func(ps_1 * alpha_k0)

            if f <= self._current_f:
                pass
            #TODO: add quad step option
            pass

        # For the subsequent line search do the following
        # (step I2 in [1])
        else:
            # Increases the step size of the previous accepted step as it
            # is only decreased in subsequent boundary manipulation.
            return ps_2 * alpha_k0

    def __very_close(self, x, y):
        '''     Returns true if x is very close in value to y.   '''
        return np.nextafter(x, y) >= y

    def obj_and_grad_func(self, alpha: float):
        '''
        For a given alpha this returns the values of the objective function
        and it's derivative.
        '''
        point_alpha = self._current + alpha * self._px
        fs_alpha = self._evaluator.evaluate([point_alpha])
        f_alpha, dfdx_alpha = fs_alpha[0]

        dfdalpha = np.matmul(np.transpose(dfdx_alpha), self._px)

        return f_alpha, dfdalpha

    # def __secant2(self, a, b):
    #     '''
    #     This function is part of the Hager-Zhang line search method [1].

    #     This function is referred to as secant^2 and described as steps
    #     S1-4 in [1].

    #     Preforms a secant step to update the bracketing interval of alpha.
    #     Given an interval that brackets a root, this procedure performs an
    #     update of both end points using two intermediate points generated
    #     using the secant interpolation `self.__secant_for_alpha()`.
    #     Assuming the interval [a, b] satisfy the opposite slope conditions.

    #     The opposite slope conditions:
    #                             φ(a) ≤ φ(0) + epsilon ,
    #                             φ'(a) < 0 (w.r.t every parameter),
    #                             φ'(b) ≥ 0 (w.r.t every parameter).
    #     where φ(α) = f(x+α*px), f() is the function binging minimised,
    #     x is the current parameter set, px is the newton direction,
    #     and alpha is the step size.

    #     :param a: power bound of alpha
    #     :param b: upper bound of alpha
    #     '''

    #     # Preforming a secant to propose a value of alpha.
    #     # (step S1 in [1])
    #     c = self.__secant_for_alpha(a,b)
    #     # CHECK IF c SATISFY THE WOLFE CONDITIONS i.e pass to tell!!!!!!!
    #     # IF IT HAS STOP SEARCHING THIS DIRECTION!!!!

    #     # IF WOLFE CONDITIONS AREAN'T MEET DO THIS thing below

    #     # (typically) updating one side of the bracketing interval
    #     print('first update secant')
    #     A, B = self.__update(a, b, c)

    #     # (typically) updating the otherside side of the bracketing interval
    #     # (step S2 in [1])
    #     if c == A:
    #         # Preforming a secant to propose a value of alpha.
    #         C = self.__secant_for_alpha (a, A)

    #     # (step S3 in [1])
    #     if c == B:
    #         # Preforming a secant to propose a value of alpha.
    #         C = self.__secant_for_alpha (b, B)

    #     # (step S4 in [1])
    #     if c == A or c == B:
    #         # CHECK IF C SATISFY THE WOLFE CONDITIONS i.e pass to tell!!!!!!!
    #         # IF IT HAS STOP SEARCHING THIS DIRECTION!!!!

    #         # IF WOLFE CONDITIONS AREAN'T MEET DO THIS thing below

    #         print('second update secant')
    #         A, B = self.__update(A, B, C)

    #     return A, B

