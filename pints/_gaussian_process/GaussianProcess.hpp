#pragma once

#include "Aboria.h"
#include "Kernels.hpp"
#include <Eigen/Core>

namespace Aboria {

ABORIA_VARIABLE(function, double, "function_value");

template <unsigned int D> class GaussianProcess {
public:
  using Particles_t =
      Particles<std::tuple<function>, D, std::vector, KdtreeNanoflann>;
  using position = typename Particles_t::position;
  using double_d = Vector<double, D>;
  using bool_d = Vector<bool, D>;
  using RawKernel_t = matern_kernel<D>;
  using Kernel_t = self_kernel<Particles_t, matern_kernel<D>>;
  using GradientSigmaKernel_t = gradient_by_sigma_kernel<Particles_t, matern_kernel<D>>;
  using GradientLengthscaleKernel_t = gradient_by_lengthscale_kernel<Particles_t, matern_kernel<D>>;
  using Map_t = Eigen::Map<Eigen::VectorXd>;
  using x_vector_t =
      Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, D>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
  using f_vector_t =
      Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 1>, 0,
                 Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
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
  GaussianProcess();

  grad_likelihood_return_t grad_likelihood();
  double likelihood();

  void set_data(x_vector_t x, f_vector_t f);

  void set_parameters(vector_parameter_t parameters);

  const unsigned int n_parameters() const { return D + 2; }

  double predict(const_vector_D_t x);

  Eigen::Vector2d predict_var(const_vector_D_t x);

protected:
  double m_lambda;
  bool m_uninitialised;
  int m_mult_buffer;
  int m_nsubdomain;
  double_d m_lengthscales;
  Particles_t m_particles;
  RawKernel_t m_kernel;
};

template <typename Op> std::array<double, 2> eigenvalue_range(const Op &B) {
  std::array<double, 2> minmax;
  // estimate eigenvalue range via n Gershgorin discs
  minmax[1] = std::numeric_limits<double>::min();
  minmax[0] = std::numeric_limits<double>::max();
  for (int i = 0; i < B.rows(); ++i) {
    const double centre = B.coeff(i, i);
    double R = 0;
    for (int j = 0; j < i; ++j) {
      R += std::abs(B.coeff(i, j));
    }
    for (int j = i + 1; j < B.cols(); ++j) {
      R += std::abs(B.coeff(i, j));
    }
    const double max_i = centre + R;
    const double min_i = centre - R;
    if (max_i > minmax[1]) {
      minmax[1] = max_i;
    }
    if (min_i < minmax[0]) {
      minmax[0] = min_i;
    }
  }
  if (minmax[0] < 0) {
    minmax[0] = 0.0;
  }

  std::cout << "eigenvalue approx range: " << minmax[0] << "-" << minmax[1]
            << std::endl;
  return minmax;
}

template <typename F>
std::vector<double>
calculate_chebyshev_coefficients(const std::vector<double> points,
                                 const Eigen::MatrixXd &polynomials,
                                 const F &f) {

  // initialise coefficients to zero
  std::vector<double> coefficients(points.size(), 0);

  // generate chebyshev interpolation coefficients for f
  const int n = points.size() - 1;
  for (int k = 0; k < n + 1; ++k) {
    coefficients[0] += f(points[k]) * polynomials(k, 0);
  }
  coefficients[0] /= n + 1;
  for (int i = 1; i < n + 1; ++i) {
    for (int k = 0; k < n + 1; ++k) {
      coefficients[i] += f(points[k]) * polynomials(k, i);
    }
    coefficients[i] *= 2.0 / (n + 1);
  }
  return coefficients;
}
} // namespace Aboria

#include "GaussianProcess.tcc"
