#include "GaussianProcess.hpp"
#include <algorithm>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace Aboria {

template <unsigned int D>
GaussianProcess<D>::GaussianProcess()
    : m_lambda(1e-5), m_uninitialised(true),m_nsubdomain(20),
      m_lengthscales(double_d::Constant(1)), m_kernel(1, m_lengthscales) {}

template <unsigned int D> double GaussianProcess<D>::likelihood() { return 0; }

template <unsigned int D>
typename GaussianProcess<D>::grad_likelihood_return_t
GaussianProcess<D>::grad_likelihood() {
  grad_likelihood_return_t gradient = grad_likelihood_return_t::Zero();
  return gradient;
}

template <unsigned int D>
void GaussianProcess<D>::set_data(x_vector_t x, f_vector_t f) {
  m_particles.resize(f.size());
  double_d min = double_d::Constant(std::numeric_limits<double>::max());
  double_d max = double_d::Constant(std::numeric_limits<double>::min());
  for (size_t i = 0; i < m_particles.size(); ++i) {
    for (size_t j = 0; j < D; ++j) {
      get<position>(m_particles)[i][j] = x(i, j);
      if (x(i, j) > max[j]) {
        max[j] = x(i, j) + 10*std::numeric_limits<double>::epsilon();
      }
      if (x(i, j) < min[j]) {
        min[j] = x(i, j);
      }
    }
    get<function>(m_particles)[i] = f(i);
  }
  m_particles.init_neighbour_search(min, max, bool_d::Constant(false),
                                    m_nsubdomain);

  m_uninitialised = true;
}

template <unsigned int D>
void GaussianProcess<D>::set_parameters(vector_parameter_t parameters) {
  for (size_t j = 0; j < D; ++j) {
    m_lengthscales[j] = parameters[j];
  }
  m_kernel.set_lengthscale(m_lengthscales);
  m_kernel.set_sigma(parameters[D]);
  m_lambda = parameters[D + 1];
  m_uninitialised = true;
}

template <unsigned int D>
double GaussianProcess<D>::predict(const_vector_D_t argx) {
  return 0;
}
template <unsigned int D>
Eigen::Vector2d GaussianProcess<D>::predict_var(const_vector_D_t argx) {
  Eigen::Vector2d mean_var = Eigen::Vector2d::Zero();
  return mean_var;
}

} // namespace Aboria
