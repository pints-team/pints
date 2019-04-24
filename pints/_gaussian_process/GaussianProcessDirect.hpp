#pragma once

#include "Aboria.h"
#include "GaussianProcess.hpp"
#include <Eigen/Core>

namespace Aboria {

template <unsigned int D>
class GaussianProcessDirect : public GaussianProcess<D> {
  using base_t = GaussianProcess<D>;
  using Particles_t = typename base_t::Particles_t;
  using Kernel_t = typename base_t::Kernel_t;
  using GradientSigmaKernel_t = typename base_t::GradientSigmaKernel_t;
  using GradientLengthscaleKernel_t =
      typename base_t::GradientLengthscaleKernel_t;
  using Map_t = Eigen::Map<Eigen::VectorXd>;

  using Solver_t = Eigen::LLT<Eigen::MatrixXd>;

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
  GaussianProcessDirect();

  grad_likelihood_return_t grad_likelihood();
  double likelihood();

  double predict(const_vector_D_t x);

  Eigen::Vector2d predict_var(const_vector_D_t x);

private:
  void initialise();

  Eigen::MatrixXd m_K;
  std::vector<Eigen::MatrixXd> m_gradKs;
  Solver_t m_solver;
  Eigen::VectorXd m_invKy;
};
} // namespace Aboria

#include "GaussianProcessDirect.tcc"
