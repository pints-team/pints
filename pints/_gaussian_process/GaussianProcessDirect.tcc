#include "GaussianProcessDirect.hpp"
#include <algorithm>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace Aboria {

template <unsigned int D>
GaussianProcessDirect<D>::GaussianProcessDirect() : base_t() {}

template <unsigned int D> void GaussianProcessDirect<D>::initialise() {
  // normalise kernel
  // const double max_eigenvalue = calculate_eigenvalue();
  // std::cout << "max eigenvalue is " << max_eigenvalue << std::endl;
  // m_kernel.set_sigma(m_kernel.get_sigma() / std::sqrt(max_eigenvalue));

  // create operators
  auto K = create_dense_operator(
      this->m_particles, this->m_particles,
      Kernel_t(this->m_kernel, std::pow(this->m_lambda, 2)));
  const int n = this->m_particles.size();
  m_K.resize(n, n);
  K.assemble(m_K);

  m_gradKs.resize(D + 1);
  for (int i = 0; i < D; ++i) {
    auto gradKs =
        create_dense_operator(this->m_particles, this->m_particles,
                              GradientLengthscaleKernel_t(this->m_kernel, i));
    m_gradKs[i].resize(n, n);
    gradKs.assemble(m_gradKs[i]);
  }

  auto gradSigmaK =
      create_dense_operator(this->m_particles, this->m_particles,
                            GradientSigmaKernel_t(this->m_kernel));
  m_gradKs[D].resize(n, n);
  gradSigmaK.assemble(m_gradKs[D]);

  // create solver
  m_solver.compute(m_K);

  Eigen::Map<Eigen::VectorXd> y(get<function>(this->m_particles).data(),
                                this->m_particles.size());
  // std::cout << "y = [" << y(0) << "," << y(1) << "," << y(2) << "..."
  //          << std::endl;
  m_invKy = m_solver.solve(y);
  // std::cout << "error = "<<(m_K*m_invKy-y).norm()<<std::endl;
  if (m_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "invKy solver failed to converge to set tolerance");
  }

  this->m_uninitialised = false;
}

template <unsigned int D> double GaussianProcessDirect<D>::likelihood() {
  if (this->m_particles.size() == 0) {
    return 0;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  // log det ( m_K )
  double ld = 0;
  auto &U = m_solver.matrixL();
  for (size_t i = 0; i < U.rows(); ++i) {
    ld += std::log(U(i, i));
  }
  ld *= 2;

  Eigen::Map<Eigen::VectorXd> y(get<function>(this->m_particles).data(),
                                this->m_particles.size());
  const double result = -0.5 * ld - 0.5 * y.dot(m_invKy);
  return result;
}

template <unsigned int D>
typename GaussianProcessDirect<D>::grad_likelihood_return_t
GaussianProcessDirect<D>::grad_likelihood() {
  grad_likelihood_return_t gradient = grad_likelihood_return_t::Zero();

  if (this->m_particles.size() == 0) {
    return gradient;
  }
  if (this->m_uninitialised) {
    initialise();
  }

  const int n = this->m_particles.size();
  for (int i = 0; i <= D; ++i) {
    gradient[i] += m_solver.solve(m_gradKs[i]).trace();
  }
  // final entry in gradient is gradient with m_lambda
  gradient[D + 1] =
      m_solver.solve(2 * this->m_lambda * Eigen::MatrixXd::Identity(n, n))
          .trace();

  // second term
  for (int i = 0; i <= D; ++i) {
    gradient[i] += 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient[D + 1] += this->m_lambda * m_invKy.dot(m_invKy);

  return gradient;
}

template <unsigned int D>
double GaussianProcessDirect<D>::predict(const_vector_D_t argx) {

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
Eigen::Vector2d GaussianProcessDirect<D>::predict_var(const_vector_D_t argx) {
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
