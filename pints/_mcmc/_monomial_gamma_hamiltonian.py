#
# Monomial gamma HMC MCMC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np

from scipy import integrate
from scipy import interpolate


class MonomialGammaHamiltonianMCMC(pints.SingleChainMCMC):
    r"""
    Implements Monomial Gamma HMC as described in [1]_ - a generalisation
    of HMC as described in [2]_ - involving a non-physical kinetic energy term.

    Uses a physical analogy of a particle moving across a landscape under
    Hamiltonian dynamics to aid efficient exploration of parameter space.
    Introduces an auxilary variable -- the momentum (``p_i``) of a particle
    moving in dimension ``i`` of negative log posterior space -- which
    supplements the position (``q_i``) of the particle in parameter space. The
    particle's motion is dictated by solutions to Hamilton's equations,

    .. math::
        dq_i/dt =   \partial H/\partial p_i,
        dp_i/dt = - \partial H/\partial q_i.

    The Hamiltonian is given by,

    .. math::
        H(q,p) =       U(q)       +        K(p)
               = -log(p(q|X)p(q)) +
                 \Sigma_{i=1}^{d} (
                 -g(p_i) + (2/c) \text{log}(1 + \text{exp}(cg(p_i))))

    where ``d`` is the dimensionality of model, ``U`` is the potential
    energy and ``K`` is the kinetic energy term. Note the kinetic energy
    is the 'soft' version described in [1], where,

    .. math::
        g(p_i) = (1 / m_i) \text{sign}|p_i|^{1 / a}

    To numerically integrate Hamilton's equations, it is essential to use a
    sympletic discretisation routine, of which the most typical approach is
    the leapfrog method,

    .. math::
        p_i(t + \epsilon/2) &= p_i(t) - (\epsilon/2) dU(q_i)/ dq_i\\
        q_i(t + \epsilon) &= q_i(t) + \epsilon d K(p_i(t + \epsilon/2))/dp_i\\
        p_i(t + \epsilon) &= p_i(t + \epsilon/2) -
                             (\epsilon/2) dU(q_i + \epsilon)/ dq_i

    The derivative of the soft kinetic energy term is given by,

    .. math::
        d K(p_i)/dp_i =
        |p_i|^{-1 + 1 / a}\text{sign}(p_i) \times
        \text{tanh}(c|p_i|^{1/a}\text{sign}(p_i) / {2 m_i}) / {a m_i}

    In particular, the algorithm we implement follows eqs. (4.14)-(4.16) in
    [2]_, since we allow different epsilon according to dimension.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Towards Unifying Hamiltonian Monte Carlo and Slice Sampling
           Yizhe Zhang, Xiangyu Wang, Changyou Chen, Ricardo Henao, Kai Fan,
           Lawrence Cari.
           Advances in Neural Information Processing Systems (NIPS)


    .. [2] MCMC using Hamiltonian dynamics
           Radford M. Neal, Chapter 5 of the Handbook of Markov Chain Monte
           Carlo by Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng.
    """
    def __init__(self, x0, sigma0=None):
        super(MonomialGammaHamiltonianMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Current point in the Markov chain
        self._current = None            # Aka current_q in the chapter
        self._current_energy = None     # Aka U(current_q) = -log_pdf
        self._current_gradient = None
        self._current_momentum = None   # Aka current_p

        # Current point in the leapfrog iterations
        self._momentum = None       # Aka p in the chapter
        self._position = None       # Aka q in the chapter
        self._gradient = None       # Aka grad_U(q) in the chapter

        # Iterations, acceptance monitoring, and leapfrog iterations
        self._mcmc_iteration = 0
        self._mcmc_acceptance = 0
        self._frog_iteration = 0

        # Default number of leapfrog iterations
        self._n_frog_iterations = 20

        # Default integration step size for leapfrog algorithm
        self._epsilon = 0.1
        self._step_size = None
        self.set_leapfrog_step_size(np.diag(self._sigma0))

        # Divergence checking
        # Create a vector of divergent iterations
        self._divergent = np.asarray([], dtype='int')

        # Default threshold for Hamiltonian divergences
        # (currently set to match Stan)
        self._hamiltonian_threshold = 10**3

        # Estimate quantities needed for sampling from the ke distribution
        self._a = 1.0
        self._c = 0.2
        self._m = 1.0

    def a(self):
        """
        Returns ``a`` in kinetic energy function.
        """
        return self._a

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._running = True
            self._initialise_ke()

        # Notes:
        #  Ask is responsible for updating the position, which is the point
        #   returned to the user
        #  Tell is then responsible for updating the momentum, which uses the
        #   gradient at this new point
        #  The MCMC step happens in tell, and does not require any new
        #   information (it uses the log_pdf and gradient of the final point
        #   in the leapfrog run).

        # Very first iteration
        if self._current is None:

            # Ask for the pdf and gradient of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # First iteration of a run of leapfrog iterations
        if self._frog_iteration == 0:

            # Sample random momentum for current point using identity cov
            self._current_momentum = self._sample_momentum()

            # First leapfrog position is the current sample in the chain
            self._position = np.array(self._current, copy=True)
            self._gradient = np.array(self._current_gradient, copy=True)
            self._momentum = np.array(self._current_momentum, copy=True)

            # Perform a half-step before starting iteration 0 below
            self._momentum -= self._scaled_epsilon * self._gradient * 0.5

        # Perform a leapfrog step for the position
        self._position += self._scaled_epsilon * (
            self._K_deriv(self._momentum, self._a, self._c, self._m)
        )

        # Ask for the pdf and gradient of the current leapfrog position
        # Using this, the leapfrog step for the momentum is performed in tell()
        self._ready_for_tell = True
        return np.array(self._position, copy=True)

    def c(self):
        """
        Returns ``c`` in kinetic energy function.
        """
        return self._c

    def _cdf(self, p, a, c, m, z):
        """
        Auxillary kinetic energy cumulative density defined in [1]_.
        """
        return integrate.quad(lambda p1: self._pdf(p1, a, c, m, z),
                              -np.inf, p)[0]

    def divergent_iterations(self):
        """
        Returns the iteration number of any divergent iterations.
        """
        return self._divergent

    def epsilon(self):
        """
        Returns epsilon used in leapfrog algorithm.
        """
        return self._epsilon

    def _g(self, p, a, m):
        """
        Helper function used to define soft kinetic energy in [1].
        """
        return (1 / m) * np.sign(p) * np.abs(p)**(1 / a)

    def hamiltonian_threshold(self):
        """
        Returns threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        return self._hamiltonian_threshold

    def _initialise_ke(self):
        """
        Initialises functions needed for sampling from soft kinetic energy
        function defined in [1]_.
        """
        z = integrate.quad(
            lambda p: np.exp(-self._K_indiv(p, self._a, self._c, self._m)),
            -np.inf, np.inf)[0]
        self._f = self._inverse_cdf_calculator(self._a, self._c,
                                               self._m, z)

    def _inverse_cdf_calculator(self, a, c, m, z, pmax=100):
        """
        Estimates empirical cdf.
        """
        p = np.linspace(-pmax, pmax, 1000)
        lcdf = [self._cdf(p[i], a, c, m, z) for i in range(1000)]
        f = interpolate.interp1d(lcdf, p, fill_value="extrapolate")
        return f

    def _K(self, v_p, a, c, m):
        """
        Soft kinetic energy function defined in [1]_.
        """
        return np.sum([self._K_indiv(p, a, c, m) for p in v_p])

    def _K_deriv(self, v_p, a, c, m):
        """
        Derivative of soft kinetic energy function defined in [1]_.
        """
        return np.array([self._K_deriv_indiv(p, a, c, m) for p in v_p])

    def _K_deriv_indiv(self, p, a, c, m):
        """
        Derivative of soft kinetic energy function defined in [1]_ for
        individual momentum component.
        """
        abs_p = np.abs(p)
        sign_p = np.sign(p)
        tanh = np.tanh(0.5 * c * abs_p**(1.0 / a) * sign_p / m)
        return abs_p**(-2 + 1.0 / a) * p * sign_p * tanh / (a * m)

    def _K_indiv(self, p, a, c, m):
        """
        Soft kinetic energy function defined in [1]_ for individual momentum
        component.
        """
        return -self._g(p, a, m) + (2.0 / c) * np.log(
            1.0 + np.exp(c * self._g(p, a, m)))

    def leapfrog_steps(self):
        """
        Returns the number of leapfrog steps to carry out for each iteration.
        """
        return self._n_frog_iterations

    def leapfrog_step_size(self):
        """
        Returns the step size for the leapfrog algorithm.
        """
        return self._step_size

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._mcmc_acceptance)

    def mass(self):
        """
        Returns mass ``m`` in kinetic energy function.
        """
        return self._m

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Monomial-Gamma Hamiltonian Monte Carlo'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 5

    def _pdf(self, p, a, c, m, z):
        """
        Auxillary kinetic energy probability density defined in [1]_.
        """
        return (1.0 / z) * np.exp(-self._K_indiv(p, a, c, m))

    def _sample_momentum(self):
        """
        Samples from soft kinetic energy pdf defined in [1]_.
        """
        us = np.random.uniform(size=self._n_parameters)
        return np.array([self._f([u])[0] for u in us])

    def scaled_epsilon(self):
        """
        Returns scaled epsilon used in leapfrog algorithm.
        """
        return self._scaled_epsilon

    def set_a(self, a):
        """
        Sets ``a`` in kinetic energy function.
        """
        if a <= 0:
            raise ValueError("a must be positive")
        self._a = a

    def set_c(self, c):
        """
        Sets ``c`` in kinetic energy function.
        """
        if c <= 0:
            raise ValueError("c must be positive")
        self._c = c

    def set_epsilon(self, epsilon):
        """
        Sets epsilon for the leapfrog algorithm.
        """
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError('epsilon must be positive for leapfrog algorithm')
        self._epsilon = epsilon
        self._set_scaled_epsilon()

    def set_hamiltonian_threshold(self, hamiltonian_threshold):
        """
        Sets threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        if hamiltonian_threshold < 0:
            raise ValueError('Threshold for divergent iterations must be ' +
                             'non-negative.')
        self._hamiltonian_threshold = hamiltonian_threshold

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[leapfrog_steps, leapfrog_step_size,
        a, c, mass]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_leapfrog_steps(x[0])
        self.set_leapfrog_step_size(x[1])
        self.set_a(x[2])
        self.set_c(x[3])
        self.set_mass(x[4])

    def set_leapfrog_steps(self, steps):
        """
        Sets the number of leapfrog steps to carry out for each iteration.
        """
        steps = int(steps)
        if steps < 1:
            raise ValueError('Number of steps must exceed 0.')
        self._n_frog_iterations = steps

    def set_leapfrog_step_size(self, step_size):
        """
        Sets the step size for the leapfrog algorithm.
        """
        a = np.atleast_1d(step_size)
        if len(a[a < 0]) > 0:
            raise ValueError(
                'Step size for leapfrog algorithm must' +
                'be greater than zero.'
            )
        if len(a) == 1:
            step_size = np.repeat(step_size, self._n_parameters)
        elif not len(step_size) == self._n_parameters:
            raise ValueError(
                'Step size should either be of length 1 or equal to the' +
                'number of parameters'
            )
        self._step_size = step_size
        self._set_scaled_epsilon()

    def set_mass(self, m):
        """
        Sets mass ``m`` in kinetic energy function.
        """
        if m <= 0:
            raise ValueError("Mass must be positive")
        self._m = m

    def _set_scaled_epsilon(self):
        """
        Rescales epsilon along the dimensions of step_size.
        """
        self._scaled_epsilon = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            self._scaled_epsilon[i] = self._epsilon * self._step_size[i]

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        energy, gradient = reply

        # Check reply, copy gradient
        energy = float(energy)
        gradient = pints.vector(gradient)
        assert gradient.shape == (self._n_parameters, )

        # Energy = -log_pdf, so flip both signs!
        energy = -energy
        gradient = -gradient

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(energy):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, energy, and gradient
            self._current = self._x0
            self._current_energy = energy
            self._current_gradient = gradient

            # Increase iteration count
            self._mcmc_iteration += 1

            # Mark current as read-only, so it can be safely returned.
            # Gradient won't be returned (only -gradient, so no need.
            self._current.setflags(write=False)

            # Return first point in chain
            return (
                self._current,
                (-self._current_energy, -self._current_gradient),
                True
            )

        # Set gradient of current leapfrog position
        self._gradient = gradient

        # Update the leapfrog iteration count
        self._frog_iteration += 1

        # Not the last iteration? Then perform a leapfrog step and return
        if self._frog_iteration < self._n_frog_iterations:
            self._momentum -= self._scaled_epsilon * self._gradient

            # Return None to indicate there is no new sample for the chain
            return None

        # Final leapfrog iteration: only do half a step
        self._momentum -= self._scaled_epsilon * self._gradient * 0.5

        # Before starting accept/reject procedure, check if the leapfrog
        # procedure has led to a finite momentum and logpdf. If not, reject.
        accept = 0
        if np.isfinite(energy) and np.all(np.isfinite(self._momentum)):

            # Evaluate potential and kinetic energies at start and end of
            # leapfrog trajectory
            current_U = self._current_energy
            current_K = self._K(self._current_momentum, self._a,
                                self._c, self._m)
            proposed_U = energy
            proposed_K = self._K(self._momentum, self._a, self._c, self._m)

            # Check for divergent iterations by testing whether the
            # Hamiltonian difference is above a threshold
            div = proposed_U + proposed_K - (self._current_energy + current_K)
            if np.abs(div) > self._hamiltonian_threshold:  # pragma: no cover
                self._divergent = np.append(
                    self._divergent, self._mcmc_iteration)
                self._momentum = self._position = self._gradient = None
                self._frog_iteration = 0

                # Update MCMC iteration count
                self._mcmc_iteration += 1

                # Update acceptance rate (only used for output!)
                self._mcmc_acceptance = (
                    (self._mcmc_iteration * self._mcmc_acceptance + accept) /
                    (self._mcmc_iteration + 1))

                return (
                    self._current,
                    (-self._current_energy, -self._current_gradient),
                    False
                )

            # Accept/reject
            else:
                r = np.exp(current_U - proposed_U + current_K - proposed_K)
                if np.random.uniform(0, 1) < r:
                    accept = 1
                    self._current = self._position
                    self._current_energy = energy
                    self._current_gradient = gradient

                    # Mark current as read-only, so it can be safely returned.
                    # Gradient won't be returned (only -gradient, so no need.
                    self._current.setflags(write=False)

        # Reset leapfrog mechanism
        self._momentum = self._position = self._gradient = None
        self._frog_iteration = 0

        # Update MCMC iteration count
        self._mcmc_iteration += 1

        # Update acceptance rate (only used for output!)
        self._mcmc_acceptance = (
            (self._mcmc_iteration * self._mcmc_acceptance + accept) /
            (self._mcmc_iteration + 1))

        # Return current position as next sample in the chain
        return (
            self._current,
            (-self._current_energy, -self._current_gradient),
            accept != 0
        )
