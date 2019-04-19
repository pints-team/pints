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
  using GradientSigmaKernel_t =
      gradient_by_sigma_kernel<Particles_t, matern_kernel<D>>;
  using GradientLengthscaleKernel_t =
      gradient_by_lengthscale_kernel<Particles_t, matern_kernel<D>>;
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
  void set_uninitialised(const bool arg) { m_uninitialised = arg; }

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

  // std::cout << "eigenvalue approx range: " << minmax[0] << "-" << minmax[1]
  //          << std::endl;
  return minmax;
}

void initialise_chebyshev(const int n, std::vector<double> &m_chebyshev_points,
                          Eigen::MatrixXd &m_chebyshev_polynomials) {

  const double pi = 3.14159265358979323846;

  // std::cout << "initialising chebyshev with n = " << n << std::endl;

  // generate chebyshev nodes
  m_chebyshev_points.resize(n + 1);
  for (int k = 0; k < n + 1; ++k) {
    m_chebyshev_points[k] = std::cos(pi * (k + 0.5) / (n + 1.0));
  }

  // generate chebyshev polynomials
  m_chebyshev_polynomials.resize(n + 1, n + 1);
  for (int k = 0; k < n + 1; ++k) {
    m_chebyshev_polynomials(k, 0) = 1.0;
    m_chebyshev_polynomials(k, 1) = m_chebyshev_points[k];
  }
  for (int j = 2; j < n + 1; ++j) {
    for (int k = 0; k < n + 1; ++k) {
      m_chebyshev_polynomials(k, j) =
          2 * m_chebyshev_points[k] * m_chebyshev_polynomials(k, j - 1) -
          m_chebyshev_polynomials(k, j - 2);
    }
  }
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

template <typename Op, typename Derived>
double calculate_gp_likelihood_chebyshev(
    Op &m_K, const Eigen::VectorXd &m_invKy,
    const Eigen::MatrixBase<Derived> &y,
    const std::vector<double> m_chebyshev_points,
    const Eigen::MatrixXd &m_chebyshev_polynomials,
    const int m_stochastic_samples_m) {

  // approximate log det with chebyshev interpolation and stochastic trace
  // estimation
  auto minmax = eigenvalue_range(m_K);
  // minmax[0] = 0;

  const double scale = 2.0 / (minmax[1] + minmax[0]);
  const double delta = (minmax[1] - minmax[0]) / (minmax[1] + minmax[0]);

  // specify function to interpolate
  auto f = [&](const double x) {
    // return std::log(1.0 - ((1.0 - 2.0 * delta) * x + 1.0) / 2.0);
    return std::log(1.0 + delta * x);
  };

  auto chebyshev_coefficients = calculate_chebyshev_coefficients(
      m_chebyshev_points, m_chebyshev_polynomials, f);

  const int n = chebyshev_coefficients.size() - 1;
  // test chebyshev coefficients for range of x \in [0,1]
  const int tn = 5;
  for (int i = 0; i < tn; ++i) {
    const double x = (1.0 / tn) * i;
    const double log_exact = f(x);

    double T0 = 1.0;
    double T1 = x;
    double log_approx =
        chebyshev_coefficients[0] + chebyshev_coefficients[1] * T1;
    double T2;
    for (int j = 2; j < n + 1; ++j) {
      T2 = 2 * x * T1 - T0;
      log_approx += chebyshev_coefficients[j] * T2;
      T0 = T1;
      T1 = T2;
    }
    // std::cout << "log approx = " << log_approx << " log exact = " <<
    // log_exact
    //          << std::endl;
  }

  double log_det = 0;
  for (int i = 0; i < m_stochastic_samples_m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(y.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    Eigen::VectorXd u = chebyshev_coefficients[0] * v;
    if (n > 1) {
      Eigen::VectorXd w0 = v;
      // Av = (scale*B - I)v = scale*B*v - v
      Eigen::VectorXd w1 = scale * (m_K * v) - v;
      u += chebyshev_coefficients[1] * w1;
      Eigen::VectorXd w2(y.size());
      for (int j = 2; j < n + 1; ++j) {
        //  2*A*w1 - w0 = 2*(scale*B - I)*w1 - w0 = 2*(scale*B*w1 - w1) - w0
        w2 = 2 * (scale * (m_K * w1) - w1) - w0;
        u += chebyshev_coefficients[j] * w2;
        w0 = w1;
        w1 = w2;
      }
    }
    log_det += v.dot(u) / m_stochastic_samples_m;
  }

  log_det -= y.size() * std::log(scale);

  return -0.5 * log_det - 0.5 * y.dot(m_invKy);
}

template <unsigned int D, typename Solver, typename GradKOps,
          typename GradSigmaOp>
Eigen::Matrix<double, D + 2, 1>
calculate_gp_grad_likelihood(const Eigen::VectorXd &m_invKy, Solver &m_solver,
                             std::vector<GradKOps> &m_gradKs,
                             GradSigmaOp &m_gradSigmaK, const double m_lambda,
                             const int m_stochastic_samples_m) {
  using grad_likelihood_return_t = Eigen::Matrix<double, D + 2, 1>;
  grad_likelihood_return_t gradient = grad_likelihood_return_t::Zero();

  std::array<Eigen::VectorXd, D + 2> guesses;
  for (int i = 0; i < D + 2; ++i) {
    guesses[i].setZero(m_invKy.size());
  }

  // trace term
  for (int i = 0; i < m_stochastic_samples_m; ++i) {
    // std::cout << "calculate_gp_grad_likelihood: iteration " << i <<
    // std::endl;
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(m_invKy.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    for (int j = 0; j < m_gradKs.size(); ++j) {
      guesses[j] = m_solver.solveWithGuess(m_gradKs[j] * v, guesses[j]);
      gradient[j] += v.dot(guesses[j]);
      //std::cout << "finished solving with " << m_solver.iterations()
      //         << " iterations" << std::endl;
    }
    guesses[D] = m_solver.solveWithGuess(m_gradSigmaK * v, guesses[D]);
    gradient[D] += v.dot(guesses[D]);
    //std::cout << "finished solving with " << m_solver.iterations()
    //         << " iterations" << std::endl;
    guesses[D + 1] =
        m_solver.solveWithGuess((2 * m_lambda) * v, guesses[D + 1]);
    gradient[D + 1] += v.dot(guesses[D + 1]);
    //std::cout << "finished solving with " << m_solver.iterations()
    //         << " iterations" << std::endl;
  }

  for (int i = 0; i < gradient.size(); ++i) {
    gradient[i] = -0.5 * gradient[i] / m_stochastic_samples_m;
  }

  // second term
  for (int i = 0; i < D; ++i) {
    gradient[i] += 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient[D] += 0.5 * m_invKy.dot(m_gradSigmaK * m_invKy);
  gradient[D + 1] += m_lambda * m_invKy.dot(m_invKy);

  return gradient;
}

template <unsigned int D, typename Derived, typename Solver, typename Op,
          typename GradKOps, typename GradSigmaOp>
Eigen::Matrix<double, D + 2, 1> calculate_gp_grad_likelihood_chebyshev(
    const Eigen::VectorXd &m_invKy, const Eigen::MatrixBase<Derived> &y,
    Solver &m_solver, Op &m_K, std::vector<GradKOps> &m_gradKs,
    GradSigmaOp &m_gradSigmaK, const double m_lambda,
    const std::vector<double> m_chebyshev_points,
    const Eigen::MatrixXd &m_chebyshev_polynomials,
    const int m_stochastic_samples_m) {

  using grad_likelihood_return_t = Eigen::Matrix<double, D + 2, 1>;
  grad_likelihood_return_t gradient = grad_likelihood_return_t::Zero();
  double likelihood = 0;

  std::array<double, D + 3> scales;
  std::array<double, D + 3> deltas;
  for (int i = 0; i < D; ++i) {
    auto minmax = eigenvalue_range(m_gradKs[i]);
    minmax[0] = 0;
    scales[i] = 1.0 / (minmax[1] + minmax[0]);
    deltas[i] = minmax[0] * scales[i];
  }

  auto minmax = eigenvalue_range(m_gradSigmaK);
  minmax[0] = 0;
  scales[D] = 1.0 / (minmax[1] + minmax[0]);
  deltas[D] = minmax[0] * scales[D];

  scales[D + 1] = 0.5 / m_lambda;
  deltas[D + 1] = 0;

  minmax = eigenvalue_range(m_K);
  minmax[0] = 0;
  scales[D + 2] = 1.0 / (minmax[1] + minmax[0]);
  deltas[D + 2] = minmax[0] * scales[D + 2];

  // specify function to interpolate
  auto f = [&](const double x) {
    // return std::log(1 - ((1 - 2 * delta) * x + 1) / 2);
    return std::log(1 - x);
  };

  auto chebyshev_coefficients = calculate_chebyshev_coefficients(
      m_chebyshev_points, m_chebyshev_polynomials, f);

  const int n = chebyshev_coefficients.size() - 1;
  for (int i = 0; i < m_stochastic_samples_m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(m_invKy.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    Eigen::VectorXd u = chebyshev_coefficients[0] * v;

    std::vector<Eigen::VectorXd> gradu(D + 2);
    for (int i = 0; i < D + 2; ++i) {
      gradu[i] = u;
    }
    if (n > 1) {
      Eigen::VectorXd w0 = v;
      Eigen::VectorXd w1 = v - scales[D + 2] * (m_K * v);
      u += chebyshev_coefficients[1] * w1;
      Eigen::VectorXd w2(v.size());

      std::vector<Eigen::VectorXd> gradw0(D + 2);
      std::vector<Eigen::VectorXd> gradw1(D + 2);
      std::vector<Eigen::VectorXd> gradw2(D + 2);
      for (int i = 0; i < D + 2; ++i) {
        gradw0[i].setZero(v.size());
        if (i == D + 1) {
          // final gradB is 2 * m_lambda * I
          gradw1[i] = -2 * scales[i] * m_lambda * v;
        } else if (i == D) {
          gradw1[i] = -scales[i] * (m_gradSigmaK * v);
        } else {
          gradw1[i] = -scales[i] * (m_gradKs[i] * v);
        }
        gradw2[i].resize(v.size());
        gradu[i] += chebyshev_coefficients[1] * gradw1[i];
      }
      for (int j = 2; j < n; ++j) {
        w2 = 2 * (w1 - scales[D + 2] * (m_K * w1)) - w0;
        u += chebyshev_coefficients[j] * w2;
        w0 = w1;
        w1 = w2;

        for (int i = 0; i < D + 2; ++i) {
          if (i == D + 1) {
            // final gradB is 2 * m_lambda * I
            gradw2[i] = 2 * (-2 * scales[i] * m_lambda * w1 + w1 -
                             2 * scales[i] * m_lambda * w1) -
                        w0;
          } else if (i == D) {

            gradw2[i] = 2 * (-scales[i] * (m_gradSigmaK * w1) + w1 -
                             scales[i] * (m_K * w1)) -
                        w0;
          } else {
            gradw2[i] = 2 * (-scales[i] * (m_gradKs[i] * w1) + w1 -
                             scales[i] * (m_K * w1)) -
                        w0;
          }
          gradu[i] += chebyshev_coefficients[j] * gradw2[i];
          gradw0[i] = gradw1[i];
          gradw1[i] = gradw2[i];
        }
      }
    }
    for (int i = 0; i < D + 2; ++i) {
      gradient[i] += v.dot(gradu[i]) / m_stochastic_samples_m;
    }
    likelihood += v.dot(u) / m_stochastic_samples_m;
  }
  for (int i = 0; i < D + 2; ++i) {
    gradient[i] -= m_invKy.size() * std::log(scales[i]);
  }

  likelihood -= m_invKy.size() * std::log(scales[D + 2]);
  likelihood = -0.5 * likelihood - 0.5 * y.dot(m_invKy);
  // std::cout << "likelihood = " << likelihood << std::endl;

  // second term
  for (int i = 0; i < D; ++i) {
    gradient[i] = -0.5 * gradient[i] + 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient[D] = -0.5 * gradient[D] + 0.5 * m_invKy.dot(m_gradSigmaK * m_invKy);
  gradient[D + 1] = -0.5 * gradient[D + 1] + m_lambda * m_invKy.dot(m_invKy);
  return gradient;
}

} // namespace Aboria

#include "GaussianProcess.tcc"
