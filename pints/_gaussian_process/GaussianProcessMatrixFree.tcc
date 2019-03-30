#include "GaussianProcessMatrixFree.hpp"
#include <algorithm>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace Aboria {

template <unsigned int D>
GaussianProcessMatrixFree<D>::GaussianProcessMatrixFree()
    : base_t(), m_mult_buffer(10),
      m_K(create_dense_operator(this->m_particles, this->m_particles,
                                Kernel_t(this->m_kernel, this->m_lambda))),
      m_gradSigmaK(
          create_dense_operator(this->m_particles, this->m_particles,
                                GradientSigmaKernel_t(this->m_kernel))),
      m_stochastic_samples_m(10), m_chebyshev_n(15) {
  set_tolerance(1e-6);
  set_max_iterations(1000);
  for (int i = 0; i < D; ++i) {
    m_gradKs.emplace_back(
        create_dense_operator(this->m_particles, this->m_particles,
                              GradientLengthscaleKernel_t(this->m_kernel, i)));
  }
  initialise_chebyshev(m_chebyshev_n);
}

template <unsigned int D> void GaussianProcessMatrixFree<D>::initialise() {
  // normalise kernel
  // const double max_eigenvalue = calculate_eigenvalue();
  // std::cout << "max eigenvalue is " << max_eigenvalue << std::endl;
  // m_kernel.set_sigma(m_kernel.get_sigma() / std::sqrt(max_eigenvalue));

  // create operators
  m_K = create_dense_operator(
      this->m_particles, this->m_particles,
      Kernel_t(this->m_kernel, std::pow(this->m_lambda, 2)));

  m_gradSigmaK = create_dense_operator(this->m_particles, this->m_particles,
                                       GradientSigmaKernel_t(this->m_kernel));

  for (int i = 0; i < D; ++i) {
    m_gradKs[i] =
        create_dense_operator(this->m_particles, this->m_particles,
                              GradientLengthscaleKernel_t(this->m_kernel, i));
  }

  // create solver
  m_solver.preconditioner().set_mult_buffer(m_mult_buffer);
  m_solver.compute(m_K);

  Eigen::Map<Eigen::VectorXd> y(get<function>(this->m_particles).data(),
                                this->m_particles.size());
  // std::cout << "y = [" << y(0) << "," << y(1) << "," << y(2) << "..."
  //          << std::endl;
  m_invKy = m_solver.solveWithGuess(y, m_invKy);
  // std::cout << "error = "<<(m_K*m_invKy-y).norm()<<std::endl;
  if (m_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "invKy solver failed to converge to set tolerance");
  }

  this->m_uninitialised = false;
}

template <unsigned int D> double GaussianProcessMatrixFree<D>::likelihood() {
  if (this->m_particles.size() == 0) {
    return 0;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  // approximate log det with chebyshev interpolation and stochastic trace
  // estimation
  auto minmax = eigenvalue_range(m_K);
  minmax[0] = 0;

  const double scale = 1.0 / (minmax[1] + minmax[0]);
  const double delta = minmax[0] * scale;

  // specify function to interpolate
  auto f = [&](const double x) {
    // return std::log(1.0 - ((1.0 - 2.0 * delta) * x + 1.0) / 2.0);
    return std::log(1.0 - x);
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
    std::cout << "log approx = " << log_approx << " log exact = " << log_exact
              << std::endl;
  }

  double log_det = 0;
  for (int i = 0; i < m_stochastic_samples_m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(this->m_particles.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    Eigen::VectorXd u = chebyshev_coefficients[0] * v;
    if (n > 1) {
      Eigen::VectorXd w0 = v;
      // Av = (I - scale*B)v = v - scale*B*v
      Eigen::VectorXd w1 = v - scale * (m_K * v);
      u += chebyshev_coefficients[1] * w1;
      Eigen::VectorXd w2(this->m_particles.size());
      for (int j = 2; j < n + 1; ++j) {
        //  2*A*w1 - w0 = 2*(I - scale*B)*w1 - w0 = 2*(w1 - scale*B*w1) - w0
        w2 = 2 * (w1 - scale * (m_K * w1)) - w0;
        u += chebyshev_coefficients[j] * w2;
        w0 = w1;
        w1 = w2;
      }
    }
    log_det += v.dot(u) / m_stochastic_samples_m;
  }

  log_det -= this->m_particles.size() * std::log(scale);

  Eigen::Map<Eigen::VectorXd> y(get<function>(this->m_particles).data(),
                                this->m_particles.size());

  return -0.5 * log_det - 0.5 * y.dot(m_invKy);
}

template <unsigned int D>
typename GaussianProcessMatrixFree<D>::grad_likelihood_return_t
GaussianProcessMatrixFree<D>::grad_likelihood() {
  grad_likelihood_return_t gradient = grad_likelihood_return_t::Zero();

  if (this->m_particles.size() == 0) {
    return gradient;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  // trace term
  for (int i = 0; i < m_stochastic_samples_m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(this->m_particles.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    for (int j = 0; j < m_gradKs.size(); ++j) {
      gradient[j] += v.dot(m_solver.solve(m_gradKs[j] * v));
    }
    gradient[D] += v.dot(m_solver.solve(m_gradSigmaK * v));
    gradient[D + 1] += v.dot(m_solver.solve((2 * this->m_lambda) * v));
  }

  for (int i = 0; i < gradient.size(); ++i) {
    gradient[i] = -0.5 * gradient[i] / m_stochastic_samples_m;
  }

  // second term
  for (int i = 0; i < D; ++i) {
    gradient[i] += 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient[D] += 0.5 * m_invKy.dot(m_gradSigmaK * m_invKy);
  gradient[D + 1] += this->m_lambda * m_invKy.dot(m_invKy);

  return gradient;
}

template <unsigned int D>
void GaussianProcessMatrixFree<D>::initialise_chebyshev(const int n) {

  const double pi = 3.14159265358979323846;

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

template <unsigned int D>
double GaussianProcessMatrixFree<D>::predict(const_vector_D_t argx) {

  const Vector<double, D> x = argx;

  if (this->m_particles.size() == 0) {
    return 0;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  double sum = 0;
  for (int i = 0; i < this->m_particles.size(); ++i) {
    sum += this->m_kernel(x, get<position_d<D>>(this->m_particles)[i]) *
           m_invKy(i);
  }
  return sum;
}

template <unsigned int D>
Eigen::Vector2d
GaussianProcessMatrixFree<D>::predict_var(const_vector_D_t argx) {
  Eigen::Vector2d mean_var = Eigen::Vector2d::Zero();
  const Vector<double, D> x = argx;

  if (this->m_particles.size() == 0) {
    return mean_var;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  Eigen::VectorXd kstar(this->m_particles.size());
  for (int i = 0; i < this->m_particles.size(); ++i) {
    kstar[i] = this->m_kernel(x, get<position_d<D>>(this->m_particles)[i]);
  }

  mean_var[0] = kstar.dot(m_invKy);

  Eigen::VectorXd invKstar = m_solver.solve(kstar);
  if (m_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "invKstar solver failed to converge to set tolerance");
  }
  mean_var[1] =
      this->m_kernel(x, x) + std::pow(this->m_lambda, 2) - kstar.dot(invKstar);
  return mean_var;
}

} // namespace Aboria
