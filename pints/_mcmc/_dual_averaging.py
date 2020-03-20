import numpy as np


class DualAveragingAdaption:
    """
    Implements a Dual Averaging scheme to adapt the step size ``epsilon``, as
    per [1]_, and estimates the (fully dense) inverse mass matrix using the
    sample covariance of the accepted parameter, as suggested in [2]_

    The adaption is done using the same windowing method employed by STAN,
    which is done over three or more windows:

    - initial window: epsilon is adapted using dual averaging
    - base window: epsilon continues to be adapted using dual averaging, this
      adaption completes at the end of this window. The inverse mass matrix is
      adaped at the end of the window by taking the sample covariance of all
      parameter points in this window.
    - terminal window: epsilon is adapted using dual averaging, which completes
      at the end of the window

    If the number of warmup steps requested by the user is greater than the sum
    of these three windows, then additional base windows are added, each with a
    size double that of the previous window

    References
    ----------
    .. [1] Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo.
           Journal of Machine Learning Research, 15(1), 1593-1623.

    .. [2] `A Conceptual Introduction to Hamiltonian Monte Carlo`,
            Michael Betancourt

    Attributes
    ----------

    inv_mass_matrix: ndarray
        inverse of the adapted mass matrix

    mass_matrix: ndarray
        the adapted mass matrix

    epsilon: float
        the adapted step size epsilon

    """

    def __init__(self, num_warmup_steps, target_accept_prob,
                 init_epsilon, init_inv_mass_matrix):
        # defaults taken from STAN
        self._initial_window = 75
        self._base_window = 25
        self._terminal_window = 50
        self._epsilon = init_epsilon
        self.set_inv_mass_matrix(np.copy(init_inv_mass_matrix))
        self._target_accept_prob = target_accept_prob

        minimum_warmup_steps = self._initial_window + self._terminal_window + \
            self._base_window

        if num_warmup_steps < minimum_warmup_steps:
            raise ValueError(
                'Number of warmup steps less than the minimum value {}'.
                format(minimum_warmup_steps)
            )

        self._warmup_steps = num_warmup_steps
        self._counter = 0
        self._next_window = self._initial_window + self._base_window
        self._adapting = True

        self.init_sample_covariance(self._base_window)
        self.init_adapt_epsilon()

    def get_inv_mass_matrix(self):
        return self._inv_mass_matrix

    def set_inv_mass_matrix(self, inv_mass_matrix):
        """
        We calculate the mass matrix whenever the inverse mass matrix is set
        """
        if inv_mass_matrix.ndim == 1:
            self._mass_matrix = 1.0 / inv_mass_matrix
            self._inv_mass_matrix = inv_mass_matrix
        else:
            try:
                self._mass_matrix = np.linalg.inv(inv_mass_matrix)
            except np.linalg.LinAlgError:
                print('WARNING: adapted mass matrix is ill-conditioned')
                return
            self._inv_mass_matrix = inv_mass_matrix

    def get_mass_matrix(self):
        return self._mass_matrix

    def get_epsilon(self):
        return self._epsilon

    def step(self, x, accept_prob):
        """
        Perform a single step of the adaption

        Arguments
        ---------

        x: ndarray
            the next accepted mcmc parameter point

        accept_prob: float
            the acceptance probability of the last NUTS/HMC mcmc step
        """

        if not self._adapting:
            return

        self._counter += 1

        if self._counter >= self._warmup_steps:
            self._epsilon = self.final_epsilon()
            self._adapting = False
            return

        self.adapt_epsilon(accept_prob)
        if self._counter > self._initial_window:
            self.add_parameter_sample(x)

        if self._counter >= self._next_window:
            self.set_inv_mass_matrix(self.calculate_sample_variance())
            if self._counter >= self._warmup_steps - self._terminal_window:
                self._next_window = self._warmup_steps
            else:
                self._base_window *= 2
                self._next_window = min(
                    self._counter + self._base_window,
                    self._warmup_steps - self._terminal_window
                )
            self.init_sample_covariance(self._next_window - self._counter)
            self._epsilon = self.final_epsilon()
            self.init_adapt_epsilon()

    def init_adapt_epsilon(self):
        """
        Start a new dual averaging adaption for epsilon
        """
        # default values taken from [1]_
        self._mu = np.log(10 * self._epsilon)
        self._log_epsilon_bar = np.log(1)
        self._H_bar = 0
        self._gamma = 0.05
        self._t0 = 10
        self._kappa = 0.75

    def adapt_epsilon(self, accept_prob):
        """
        Perform a single step of the dual averaging scheme
        """
        self._H_bar = (1 - 1.0 / (self._counter + self._t0)) * self._H_bar \
            + 1.0 / (self._counter + self._t0) * \
            (self._target_accept_prob - accept_prob)
        self._log_epsilon = self._mu  \
            - (np.sqrt(self._counter) / self._gamma) \
            * self._H_bar
        self._log_epsilon_bar = self._counter**(-self._kappa) \
            * self._log_epsilon + \
            (1 - self._counter**(-self._kappa)) * self._log_epsilon_bar
        self._epsilon = np.exp(self._log_epsilon)

    def final_epsilon(self):
        """
        Perform the final step of the dual averaging scheme
        """
        return np.exp(self._log_epsilon_bar)

    def init_sample_covariance(self, size):
        """
        Start a new adaption window for the inverse mass matrix
        """
        n = self._inv_mass_matrix.shape[0]
        self._samples = np.empty((n, size))
        self._num_samples = 0

    def add_parameter_sample(self, x):
        """
        Store the parameter samples so that we can later on calculate a sample
        covariance
        """
        self._samples[:, self._num_samples] = x
        self._num_samples += 1

    def calculate_sample_variance(self):
        """
        Return the sample covariance of all the stored samples
        """
        assert self._num_samples == self._samples.shape[1]

        if self._inv_mass_matrix.ndim == 1:
            return np.var(self._samples, axis=1)
        else:
            return np.cov(self._samples)
