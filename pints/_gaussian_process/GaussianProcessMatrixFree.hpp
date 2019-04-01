#pragma once

#include "Aboria.h"
#include "GaussianProcess.hpp"
#include <Eigen/Core>

namespace Aboria {

template <unsigned int D>
class GaussianProcessMatrixFree : public GaussianProcess<D> {
  using base_t = GaussianProcess<D>;
  using Particles_t = typename base_t::Particles_t;
  using Kernel_t = typename base_t::Kernel_t;
  using GradientSigmaKernel_t = typename base_t::GradientSigmaKernel_t;
  using GradientLengthscaleKernel_t =
      typename base_t::GradientLengthscaleKernel_t;
  using OperatorKernel_t = KernelDense<Particles_t, Particles_t, Kernel_t>;
  using GradientSigmaOperatorKernel_t =
      KernelDense<Particles_t, Particles_t, GradientSigmaKernel_t>;
  using GradientLengthscaleOperatorKernel_t =
      KernelDense<Particles_t, Particles_t, GradientLengthscaleKernel_t>;
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
  GaussianProcessMatrixFree();

  grad_likelihood_return_t grad_likelihood();
  double likelihood();

  void set_max_iterations(const double n) { m_solver.setMaxIterations(n); }

  void set_tolerance(const double tol) { m_solver.setTolerance(tol); }

  void set_stochastic_samples(const int iterations) {
    m_stochastic_samples_m = iterations;
  }

  void set_chebyshev_n(const int n) {
    m_chebyshev_n = n;
    initialise_chebyshev(n, m_chebyshev_points, m_chebyshev_polynomials);
  }

  double predict(const_vector_D_t x);

  Eigen::Vector2d predict_var(const_vector_D_t x);

private:
  void initialise();

  double calculate_max_eigenvalue();

  int m_mult_buffer;
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

#include "GaussianProcessMatrixFree.tcc"
