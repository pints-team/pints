#include "GaussianProcess.hpp"
#include <algorithm>
#include <pybind11/pybind11.h>

namespace Aboria {

template <unsigned int D>
typename GaussianProcess<D>::gradient_t
GaussianProcess<D>::likelihood_gradient() {
  gradient_t gradient = gradient_t::Zero();
  if (m_particles.size() == 0) {
    return gradient;
  }
  if (m_uninitialised) {
    m_K = create_dense_operator(m_particles, m_particles,
                                Kernel_t(m_kernel, m_lambda));

    m_gradK = create_dense_operator(m_particles, m_particles,
                                    GradientKernel_t(m_kernel));

    m_solver.preconditioner().set_mult_buffer(m_mult_buffer);
    m_solver.compute(m_K);
  }
  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());

  // trace term
  const int Nr = 4;
  Eigen::VectorXd tmp;
  for (int i = 0; i < Nr; ++i) {
    Eigen::VectorXd r = Eigen::VectorXd::Random();
    std::transform(r.data(), r.data() + r.size(), r.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    m_invKr = m_solver.solveWithGuess(r, m_invKr);
    if (m_solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "invKr solver failed to converge to set tolerance");
    }
    std::cout << "invKr: Solver finished with " << m_solver.iterations()
              << " iterations and " << m_solver.error() << " error"
              << std::endl;
    tmp = m_gradK * r;
    for (int j = 0; j < D + 1; ++j) {
      gradient[j] += m_invKr.dot(tmp(Eigen::seq(0, Eigen::last, D + 1)));
    }
  }
  gradient /= Nr;

  // second term
  m_invKy = m_solver.solveWithGuess(y, m_invKy);
  if (m_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "invKy solver failed to converge to set tolerance");
  }
  std::cout << "invKy: Solver finished with " << m_solver.iterations()
            << " iterations and " << m_solver.error() << " error" << std::endl;
  tmp = m_gradK * m_invKy;
  for (int j = 0; j < D + 1; ++j) {
    gradient[j] -= 0.5 * m_invKy.dot(tmp(Eigen::seq(0, Eigen::last, D + 1)));
  }
  return gradient;
}

template <unsigned int D>
void GaussianProcess<D>::set_data(Eigen::Ref<Eigen::VectorXd> x,
                                  Eigen::Ref<Eigen::VectorXd> f) {
  m_particles.resize(x.size());
  double_d min = double_d::Constant(std::numeric_limits<double>::max());
  double_d max = double_d::Constant(std::numeric_limits<double>::min());
  for (size_t i = 0; i < m_particles.size(); ++i) {
    for (size_t j = 0; j < D; ++j) {
      get<position>(m_particles)[i][j] = x(i, j);
      if (x(i, j) > max[j]) {
        max[j] = x(i, j);
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
void GaussianProcess<D>::set_lengthscale(Eigen::Ref<Eigen::VectorXd> sigma) {
  for (size_t j = 0; j < D; ++j) {
    m_lengthscales[j] = sigma[j];
  }
  m_kernel.set_lengthscale(m_lengthscales);
  m_uninitialised = true;
}
} // namespace Aboria

namespace py = pybind11;
using namespace Aboria;

PYBIND11_MODULE(aboria_wrapper, m) {
  py::class_<GaussianProcess<2>>(m, "GaussianProcess2")
      .def("likelihood_gradient", &GaussianProcess<2>::likelihood_gradient)
      .def("set_data", &GaussianProcess<2>::set_data)
      .def("set_lengthscale", &GaussianProcess<2>::set_lengthscale)
      .def("set_sigma", &GaussianProcess<2>::set_sigma)
      .def("set_max_iterations", &GaussianProcess<2>::set_max_iterations)
      .def("set_tolerance", &GaussianProcess<2>::set_tolerance)
      .def("set_trace_iterations", &GaussianProcess<2>::set_trace_iterations)
      .def("set_noise", &GaussianProcess<2>::set_noise);
}
