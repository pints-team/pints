#include "GaussianProcessDenseMatrix.hpp"
#include <algorithm>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace Aboria {

template <unsigned int D>
GaussianProcessDenseMatrix<D>::GaussianProcessDenseMatrix()
    : base_t(), m_mult_buffer(10),
      m_K(create_matrix_operator(this->m_particles, this->m_particles,
                                 Kernel_t(this->m_kernel, this->m_lambda))),
      m_gradSigmaK(
          create_matrix_operator(this->m_particles, this->m_particles,
                                 GradientSigmaKernel_t(this->m_kernel))),
      m_stochastic_samples_m(10), m_chebyshev_n(15) {
  set_tolerance(1e-6);
  set_max_iterations(1000);
  for (int i = 0; i < D; ++i) {
    m_gradKs.emplace_back(
        create_matrix_operator(this->m_particles, this->m_particles,
                               GradientLengthscaleKernel_t(this->m_kernel, i)));
  }
  initialise_chebyshev(m_chebyshev_n, m_chebyshev_points,
                       m_chebyshev_polynomials);
}

template <unsigned int D> void GaussianProcessDenseMatrix<D>::initialise() {
  // normalise kernel
  // const double max_eigenvalue = calculate_eigenvalue();
  // std::cout << "max eigenvalue is " << max_eigenvalue << std::endl;
  // m_kernel.set_sigma(m_kernel.get_sigma() / std::sqrt(max_eigenvalue));

  // create operators
  m_K = create_matrix_operator(
      this->m_particles, this->m_particles,
      Kernel_t(this->m_kernel, std::pow(this->m_lambda, 2)));

  m_gradSigmaK = create_matrix_operator(this->m_particles, this->m_particles,
                                        GradientSigmaKernel_t(this->m_kernel));

  for (int i = 0; i < D; ++i) {
    m_gradKs[i] =
        create_matrix_operator(this->m_particles, this->m_particles,
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

template <unsigned int D> double GaussianProcessDenseMatrix<D>::likelihood() {
  if (this->m_particles.size() == 0) {
    return 0;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  Eigen::Map<Eigen::VectorXd> y(get<function>(this->m_particles).data(),
                                this->m_particles.size());

  return calculate_gp_likelihood_chebyshev(m_K, m_invKy, y, m_chebyshev_points,
                                           m_chebyshev_polynomials,
                                           m_stochastic_samples_m);
}

template <unsigned int D>
typename GaussianProcessDenseMatrix<D>::grad_likelihood_return_t
GaussianProcessDenseMatrix<D>::grad_likelihood() {
  grad_likelihood_return_t gradient = grad_likelihood_return_t::Zero();

  if (this->m_particles.size() == 0) {
    return gradient;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  return calculate_gp_grad_likelihood<D>(m_invKy, m_solver, m_gradKs, m_gradSigmaK,
                                      this->m_lambda, m_stochastic_samples_m);
}

template <unsigned int D>
double GaussianProcessDenseMatrix<D>::predict(const_vector_D_t argx) {

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
GaussianProcessDenseMatrix<D>::predict_var(const_vector_D_t argx) {
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
