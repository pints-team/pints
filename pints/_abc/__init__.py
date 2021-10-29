

class ABCSampler(pints.Loggable, pints.TunableMethod):
    """
    Bla bla bla
    """

    def name(self):
        """
        Something Something
        """
        raise NotImplementedError

    def ask(self):
        """
        Something Something
        """
        raise NotImplementedError

    def tell(self, x):
        """
        Something Something
        """
        raise NotImplementedError


# First do the interface i guess
class ABCController(object):
    """
    Explanations
    """


    def set_log_interval(self, iters=20, warm_up=3):
        iters = int(iters)
        if iters < 1:
            raise ValueError("Interval must be greater than 0")

        warm_up = max(0, int(warm_up))
        self._message_interval = iters
        self._message_warm_up = warm_up

    def set_log_to_file(self, filename=None, csv=False):
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False

    def set_log_to_screen(self, enabled):
        self._log_to_screen = True if enabled else False

    def max_iterations(self):
        return self._max_iterations

    def n_target(self):
        return self._n_target

    def parallel(self):
        return self._n_workers if self._parallel else False

    def run(self):
        print("main logic here")
    
    def log_filename(self):
        return self._log_filename

    def sampler(self):
        return self._sampler

    def set_max_iterations(self, iterations=10000):
        if iterations is None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_nr_samples(self, n_samples=500):
        self._n_samples = n_samples

    def set_parallel(self, parallel=False):
        if parallel is True:
            self._n_workers = int(parallel)
            self._parallel = True

        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else
            self._parallel = False
            self._n_workers = 1
