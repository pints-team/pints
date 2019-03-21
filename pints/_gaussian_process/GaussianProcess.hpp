#include "Aboria.h"
#include <Eigen/Core>

namespace Aboria {

ABORIA_VARIABLE(function, double, "function_value");

template <unsigned int D> struct matern_kernel {
  using gradient_t = Eigen::Matrix<double, D, 1>;
  using double_d = Vector<double, D>;
  double_d m_lengthscale;
  double m_sigma2;
  double m_sigma;
  static constexpr const char *m_name = "matern";
  matern_kernel(const double sigma, const double_d &lengthscales) {
    set_sigma(sigma);
    set_lengthscale(lengthscales);
  }
  void set_sigma(const double sigma) {
    m_sigma = sigma;
    m_sigma2 = std::pow(sigma, 2);
  }
  double get_sigma() { return m_sigma; }
  void set_lengthscale(const double_d &lengthscale) {
    m_lengthscale = 1.0 / lengthscale;
  }
  double operator()(const Vector<double, D> &a,
                    const Vector<double, D> &b) const {
    const double r = ((b - a) * m_lengthscale).norm();
    return m_sigma2 * (1.0 + std::sqrt(3.0) * r) *
           std::exp(-std::sqrt(3.0) * r);
  }

  gradient_t gradient(const Vector<double, D> &a,
                      const Vector<double, D> &b) const {
    gradient_t grad;
    const double_d dx2 = ((b - a) * m_lengthscale).pow(2);
    const double r = std::sqrt(dx2.sum());
    const double exp_term = std::exp(-std::sqrt(3.0) * r);
    // grad[0] = 2 * m_sigma * (1.0 + std::sqrt(3.0) * r) * exp_term;
    const double factor = 3 * m_sigma2 * exp_term;
    for (int i = 0; i < D; ++i) {
      grad[i] = m_lengthscale[i] * dx2[i] * factor;
    }
    return grad;
  }

  double gradient_by(const Vector<double, D> &a, const Vector<double, D> &b,
                     const int i) const {
    gradient_t grad;
    const double_d dx2 = ((b - a) * m_lengthscale).pow(2);
    const double r = std::sqrt(dx2.sum());
    const double exp_term = std::exp(-std::sqrt(3.0) * r);
    // grad[0] = 2 * m_sigma * (1.0 + std::sqrt(3.0) * r) * exp_term;
    const double factor = 3 * m_sigma2 * exp_term;
    return m_lengthscale[i] * dx2[i] * factor;
  }
};

template <typename Particles, typename Kernel> struct self_kernel {
  using raw_const_reference = typename Particles::raw_const_reference;
  using position = typename Particles::position;
  Kernel m_kernel;
  double m_jitter;

  self_kernel(const Kernel &kernel, const double jitter)
      : m_kernel(kernel), m_jitter(jitter) {}

  template <typename OldKernel> self_kernel(const OldKernel &kernel) {
    m_kernel.m_scale = kernel.m_kernel.m_scale;
    m_kernel.m_sigma = kernel.m_kernel.m_sigma;
    m_jitter = kernel.m_jitter;
  }

  double operator()(raw_const_reference a, raw_const_reference b) const {
    double ret = m_kernel(get<position>(a), get<position>(b));
    if (get<id>(a) == get<id>(b)) {
      ret += m_jitter;
    }
    return ret;
  }
};

template <typename Particles, typename Kernel> struct gradient_by_kernel {
  using raw_const_reference = typename Particles::raw_const_reference;
  using position = typename Particles::position;
  Kernel m_kernel;
  int m_i;

  gradient_by_kernel(const Kernel &kernel, const int i)
      : m_kernel(kernel), m_i(i) {}

  template <typename OldKernel> gradient_by_kernel(const OldKernel &kernel) {
    m_kernel.m_scale = kernel.m_kernel.m_scale;
    m_kernel.m_sigma = kernel.m_kernel.m_sigma;
    m_i = kernel.m_i;
  }

  double operator()(raw_const_reference a, raw_const_reference b) const {
    double ret = m_kernel.gradient_by(get<position>(a), get<position>(b), m_i);
    if (get<id>(a) == get<id>(b)) {
      ret += 1e-5;
    }
    return ret;
  }
};

template <unsigned int D> class GaussianProcess {
  using Particles_t =
      Particles<std::tuple<function>, D, std::vector, KdtreeNanoflann>;
  using position = typename Particles_t::position;
  using double_d = Vector<double, D>;
  using gradient_t = Eigen::Matrix<double, D+1, 1>;
  using bool_d = Vector<bool, D>;
  using RawKernel_t = matern_kernel<D>;
  using Kernel_t = self_kernel<Particles_t, matern_kernel<D>>;
  using GradientKernel_t = gradient_by_kernel<Particles_t, matern_kernel<D>>;
  using OperatorKernel_t = KernelDense<Particles_t, Particles_t, Kernel_t>;
  using GradientOperatorKernel_t =
      KernelDense<Particles_t, Particles_t, GradientKernel_t>;
  using Operator_t = MatrixReplacement<1, 1, std::tuple<OperatorKernel_t>>;
  using GradientOperator_t =
      MatrixReplacement<1, 1, std::tuple<GradientOperatorKernel_t>>;
  using Solver_t = Eigen::ConjugateGradient<
      Operator_t, Eigen::Lower | Eigen::Upper,
      MultiGridPreconditioner<Operator_t, Eigen::LLT<Eigen::MatrixXd>>>;
  using Map_t = Eigen::Map<Eigen::VectorXd>;
  using x_vector_t =
      Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, D>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  using f_vector_t =
      Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 1>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  using lengthscale_vector_t =
      Eigen::Ref<const Eigen::Matrix<double, D, 1>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

public:
  GaussianProcess();
  void likelihoodS1(double &likelihood, Eigen::Ref<gradient_t> gradient);
  double likelihood();

  void set_data(x_vector_t x, f_vector_t f);
  void set_lengthscale(lengthscale_vector_t sigma);

  void set_noise(const double lambda) {
    m_lambda = lambda;
    m_uninitialised = true;
  }

  const unsigned int n_parameters() const { return D+1; }

  void set_max_iterations(const double n) { m_solver.setMaxIterations(n); }

  void set_tolerance(const double tol) { m_solver.setTolerance(tol); }

  void set_trace_iterations(const int iterations) {
    m_trace_iterations = iterations;
  }

private:
  void initialise();
  double calculate_max_eigenvalue();

  void calculate_chebyshev_coefficients(const int n);

  template <typename Operator>
  double calculate_log_det(const Operator &A, const int m);

  template <typename Operator>
  double calculate_log_det_exact(const Operator &A);

  double m_lambda;
  bool m_uninitialised;
  int m_mult_buffer;
  int m_nsubdomain;
  double_d m_lengthscales;
  Particles_t m_particles;
  RawKernel_t m_kernel;
  Operator_t m_K;
  std::vector<GradientOperator_t> m_gradKs;
  Solver_t m_solver;
  Eigen::VectorXd m_invKy;
  Eigen::VectorXd m_invKr;
  int m_trace_iterations;
  Eigen::VectorXd m_chebyshev_coefficients;

  std::vector<double> m_chebyshev_points;
  Eigen::MatrixXd m_chebyshev_polynomials;
};
} // namespace Aboria
