#pragma once

#include "Aboria.h"
#include "GaussianProcess.hpp"
#include <Eigen/Core>

namespace Aboria {

template <unsigned int D>
class GaussianProcessH2Matrix : public GaussianProcess<D> {
  using base_t = GaussianProcess<D>;
  using Particles_t = typename base_t::Particles_t;
  using RawKernel_t = typename base_t::RawKernel_t;
  using Kernel_t = typename base_t::Kernel_t;
  using GradientSigmaKernel_t = typename base_t::GradientSigmaKernel_t;
  using GradientLengthscaleKernel_t =
      typename base_t::GradientLengthscaleKernel_t;
  using GradientSigmaRawKernel_t =
      gradient_by_sigma_raw_kernel<D, matern_kernel<D>>;
  using GradientLengthscaleRawKernel_t =
      gradient_by_lengthscale_raw_kernel<D, matern_kernel<D>>;
  using OperatorKernel_t =
      KernelH2<Particles_t, Particles_t, RawKernel_t, Kernel_t>;
  using GradientSigmaOperatorKernel_t =
      KernelH2<Particles_t, Particles_t, GradientSigmaRawKernel_t,
               GradientSigmaKernel_t>;
  using GradientLengthscaleOperatorKernel_t =
      KernelH2<Particles_t, Particles_t, GradientLengthscaleRawKernel_t,
               GradientLengthscaleKernel_t>;
  using Operator_t = MatrixReplacement<1, 1, std::tuple<OperatorKernel_t>>;
  using GradientSigmaOperator_t =
      MatrixReplacement<1, 1, std::tuple<GradientSigmaOperatorKernel_t>>;
  using GradientLengthscaleOperator_t =
      MatrixReplacement<1, 1, std::tuple<GradientLengthscaleOperatorKernel_t>>;
  using Solver_t = Eigen::ConjugateGradient<
      Operator_t, Eigen::Lower | Eigen::Upper,
      MultiGridPreconditioner<Operator_t, Eigen::LLT<Eigen::MatrixXd>>>;
  using Map_t = Eigen::Map<Eigen::VectorXd>;

  using const_vector_D_t =
      Eigen::Ref<const Eigen::Matrix<double, D, 1>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  using vector_D_t = Eigen::Ref<Eigen::Matrix<double, D, 1>, 0,
                                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  using vector_parameter_t =
      Eigen::Ref<const Eigen::Matrix<double, D + 2, 1>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

  using grad_likelihood_return_t = Eigen::Matrix<double, D + 2, 1>;

public:
  GaussianProcessH2Matrix();

  grad_likelihood_return_t grad_likelihood();
  double likelihood();

  void print_h2_errors();

  void set_max_iterations(const double n) { m_solver.setMaxIterations(n); }

  void set_tolerance(const double tol) { m_solver.setTolerance(tol); }

  void set_stochastic_samples(const int iterations) {
    m_stochastic_samples_m = iterations;
  }

  void set_chebyshev_n(const int n) {
    m_chebyshev_n = n;
    initialise_chebyshev(n, m_chebyshev_points, m_chebyshev_polynomials);
  }

  void set_h2_order(const int n) {
    m_h2_order = n;
    this->m_uninitialised = true;
  }

  double predict(const_vector_D_t x);

  Eigen::Vector2d predict_var(const_vector_D_t x);

private:
  void initialise();

  double calculate_max_eigenvalue();

  int m_mult_buffer;
  int m_h2_order;
  Operator_t m_K;
  GradientSigmaOperator_t m_gradSigmaK;
  std::vector<GradientLengthscaleOperator_t> m_gradKs;
  Solver_t m_solver;
  Eigen::VectorXd m_invKy;
  Eigen::VectorXd m_invKr;
  Eigen::VectorXd m_chebyshev_coefficients;
  int m_chebyshev_n;
  int m_stochastic_samples_m;

  std::vector<double> m_chebyshev_points;
  Eigen::MatrixXd m_chebyshev_polynomials;
};
} // namespace Aboria

#include "GaussianProcessH2Matrix.tcc"
