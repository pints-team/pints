#include "GaussianProcess.hpp"
#include <algorithm>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace Aboria {

template <unsigned int D>
GaussianProcess<D>::GaussianProcess()
    : m_lambda(1e-5), m_uninitialised(true), m_mult_buffer(10),
      m_nsubdomain(20), m_lengthscales(double_d::Constant(1)),
      m_kernel(1, m_lengthscales),
      m_K(create_dense_operator(m_particles, m_particles,
                                Kernel_t(m_kernel, m_lambda))),
      m_trace_iterations(4) {
  set_tolerance(1e-6);
  set_max_iterations(1000);
  for (int i = 0; i < D; ++i) {
    m_gradKs.emplace_back(create_dense_operator(m_particles, m_particles,
                                                GradientKernel_t(m_kernel, i)));
  }
  calculate_chebyshev_coefficients(15);
}

template <unsigned int D> void GaussianProcess<D>::initialise() {
  // normalise kernel
  // const double max_eigenvalue = calculate_eigenvalue();
  // std::cout << "max eigenvalue is " << max_eigenvalue << std::endl;
  // m_kernel.set_sigma(m_kernel.get_sigma() / std::sqrt(max_eigenvalue));

  // create operators
  m_K = create_dense_operator(m_particles, m_particles,
                              Kernel_t(m_kernel, m_lambda));

  for (int i = 0; i < D; ++i) {
    m_gradKs[i] = create_dense_operator(m_particles, m_particles,
                                        GradientKernel_t(m_kernel, i));
  }

  // create solver
  m_solver.preconditioner().set_mult_buffer(m_mult_buffer);
  m_solver.compute(m_K);

  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());
  // std::cout << "y = [" << y(0) << "," << y(1) << "," << y(2) << "..."
  //          << std::endl;
  m_invKy = m_solver.solveWithGuess(y, m_invKy);
  if (m_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "invKy solver failed to converge to set tolerance");
  }
}

template <unsigned int D> double GaussianProcess<D>::likelihood() {
  if (m_particles.size() == 0) {
    return 0;
  }
  if (m_uninitialised) {
    initialise();
  }

  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());

  // const double result =
  //    -0.5 * calculate_log_det(m_K, 10) - 0.5 * y.dot(m_invKy);
  const double result =
      -0.5 * calculate_log_det_exact(m_K) - 0.5 * y.dot(m_invKy);
  return result;
}

template <unsigned int D>
void GaussianProcess<D>::likelihoodS1(double &likelihood,
                                      Eigen::Ref<gradient_t> gradient) {
  gradient = gradient_t::Zero();
  likelihood = 0;
  if (m_particles.size() == 0) {
    return;
  }
  if (m_uninitialised) {
    initialise();
  }

  // trace term
  for (int i = 0; i < D; ++i) {
    gradient[i] -= 0.5 * calculate_log_det(m_gradKs[i], 10);
    // std::cout << "log_det estimate = " << gradient[i] << ". log_det exact = "
    //          << -0.5 * calculate_log_det_exact(m_gradKs[i]) << std::endl;
  }
  gradient[D] -= 0.5 * std::log(m_lambda);

  // second term
  for (int i = 0; i < D; ++i) {
    gradient[i] += 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient[D] += 0.5 * m_lambda * m_invKy.dot(m_invKy);

  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());

  likelihood = -0.5 * calculate_log_det(m_K, 10) - 0.5 * y.dot(m_invKy);
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

  // zero previous guesses
  m_invKr.setZero(m_particles.size());
  m_invKy.setZero(m_particles.size());

  // scale kernel by number of particles
  m_kernel.set_sigma(1.0 / m_particles.size());

  m_uninitialised = true;
}

template <unsigned int D>
void GaussianProcess<D>::set_lengthscale(lengthscale_vector_t sigma) {
  for (size_t j = 0; j < D; ++j) {
    m_lengthscales[j] = sigma[j];
  }
  m_kernel.set_lengthscale(m_lengthscales);

  m_uninitialised = true;
}

template <unsigned int D>
double GaussianProcess<D>::calculate_max_eigenvalue() {
  double max_eigenvalue = 0;
  auto K = Kernel_t(m_kernel, m_lambda);

  // n Gershgorin discs
  for (int i = 0; i < m_particles.size(); ++i) {
    const double centre = K(m_particles[i], m_particles[i]);
    double R = 0;
    for (int j = 0; j < i; ++j) {
      R += std::abs(K(m_particles[i], m_particles[j]));
    }
    for (int j = i + 1; j < m_particles.size(); ++j) {
      R += std::abs(K(m_particles[i], m_particles[j]));
    }
    const double max_i = centre + R;
    if (max_i > max_eigenvalue) {
      max_eigenvalue = max_i;
    }
  }

  Eigen::MatrixXd M(m_particles.size(), m_particles.size());
  for (int i = 0; i < m_particles.size(); ++i) {
    for (int j = 0; j < m_particles.size(); ++j) {
      M(i, j) = K(m_particles[i], m_particles[j]);
    }
  }
  Eigen::VectorXd eigenvalues = M.eigenvalues().cwiseAbs();
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();
  for (int i = 0; i < eigenvalues.size(); ++i) {
    if (eigenvalues[i] < min) {
      min = eigenvalues[i];
    }
    if (eigenvalues[i] > max) {
      max = eigenvalues[i];
    }
  }
  std::cout << "eigenvalues range from " << min << "-" << max << std::endl;

  return max_eigenvalue;
}

template <unsigned int D>
void GaussianProcess<D>::calculate_chebyshev_coefficients(const int n) {

  const double pi = 3.14159265358979323846;

  // generate chebyshev nodes
  m_chebyshev_points.resize(n);
  for (int k = 0; k < n; ++k) {
    m_chebyshev_points[k] = std::cos(pi * (k + 0.5) / (n + 1));
  }

  // generate chebyshev polynomials
  m_chebyshev_polynomials.resize(n, n);
  for (int k = 0; k < n; ++k) {
    m_chebyshev_polynomials(k, 0) = 1.0;
    m_chebyshev_polynomials(k, 1) = m_chebyshev_points[k];
  }
  for (int j = 2; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      m_chebyshev_polynomials(k, j) =
          2 * m_chebyshev_points[k] * m_chebyshev_polynomials(k, j - 1) -
          m_chebyshev_polynomials(k, j - 2);
    }
  }
}

template <unsigned int D>
template <typename Operator>
double GaussianProcess<D>::calculate_log_det(const Operator &B, const int m) {
  // estimate eigenvalue range via n Gershgorin discs
  double max_eigenvalue = std::numeric_limits<double>::min();
  double min_eigenvalue = std::numeric_limits<double>::max();
  for (int i = 0; i < m_particles.size(); ++i) {
    const double centre = B.coeff(i, i);
    double R = 0;
    for (int j = 0; j < i; ++j) {
      R += std::abs(B.coeff(i, j));
    }
    for (int j = i + 1; j < m_particles.size(); ++j) {
      R += std::abs(B.coeff(i, j));
    }
    const double max_i = centre + R;
    const double min_i = centre - R;
    if (max_i > max_eigenvalue) {
      max_eigenvalue = max_i;
    }
    if (min_i < min_eigenvalue) {
      min_eigenvalue = min_i;
    }
  }
  if (min_eigenvalue < 0) {
    min_eigenvalue = 0.0;
  }

  // std::cout << "eigen values range from " << min_eigenvalue << "-"
  //          << max_eigenvalue << std::endl;

  const double scale = 1.0 / (max_eigenvalue + min_eigenvalue);
  const double delta = min_eigenvalue * scale;

  // Eigen::MatrixXd M(B.rows(), B.cols());
  // for (int i = 0; i < B.rows(); ++i) {
  //  for (int j = 0; j < B.cols(); ++j) {
  //    M(i, j) = scale * B.coeff(i, j);
  //  }
  //}
  // Eigen::VectorXd eigenvalues = M.eigenvalues().cwiseAbs();
  // double min = std::numeric_limits<double>::max();
  // double max = std::numeric_limits<double>::min();
  // for (int i = 0; i < eigenvalues.size(); ++i) {
  //  if (eigenvalues[i] < min) {
  //    min = eigenvalues[i];
  //  }
  //  if (eigenvalues[i] > max) {
  //    max = eigenvalues[i];
  //  }
  //}
  // std::cout << "exact: eigenvalues range from " << min << "-" << max
  //          << std::endl;

  // specify function to interpolate
  auto f = [&](const double x) {
    return std::log(1 - ((1 - 2 * delta) * x + 1) / 2);
  };

  // generate chebyshev interpolation coefficients for f
  const int n = m_chebyshev_points.size();
  m_chebyshev_coefficients = Eigen::VectorXd::Zero(n);
  for (int k = 0; k < n; ++k) {
    m_chebyshev_coefficients(0) +=
        f(m_chebyshev_points[k]) * m_chebyshev_polynomials(k, 0);
  }
  m_chebyshev_coefficients(0) /= n + 1;
  for (int i = 1; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      m_chebyshev_coefficients(i) +=
          f(m_chebyshev_points[k]) * m_chebyshev_polynomials(k, i);
    }
    m_chebyshev_coefficients(i) *= 2.0 / (n + 1);
  }

  double gamma = 0;
  for (int i = 0; i < m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(m_particles.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    Eigen::VectorXd u = m_chebyshev_coefficients[0] * v;
    if (n > 1) {
      Eigen::VectorXd w0 = v;
      Eigen::VectorXd w1 = v - scale * (B * v);
      u += m_chebyshev_coefficients[1] * w1;
      Eigen::VectorXd w2(m_particles.size());
      for (int j = 2; j < n; ++j) {
        w2 = 2 * (w1 - scale * (B * w1)) - w0;
        u += m_chebyshev_coefficients[j] * w2;
        w0 = w1;
        w1 = w2;
      }
    }
    gamma += v.dot(u) / m;
  }

  return gamma + m_particles.size() * std::log(max_eigenvalue + min_eigenvalue);
}

template <unsigned int D>
template <typename Operator>
double GaussianProcess<D>::calculate_log_det_exact(const Operator &B) {
  Eigen::MatrixXd M(B.rows(), B.cols());
  B.assemble(M);

  double ld = 0;
  Eigen::LLT<Eigen::MatrixXd> chol(M);
  auto &U = chol.matrixL();
  for (size_t i = 0; i < M.rows(); ++i) {
    ld += std::log(U(i, i));
  }
  ld *= 2;
  return ld;
}

} // namespace Aboria

namespace py = pybind11;
using namespace Aboria;

PYBIND11_MODULE(gaussian_process, m) {
  py::class_<GaussianProcess<2>>(m, "GaussianProcess2")
      .def(py::init<>())
      .def("likelihoodS1", &GaussianProcess<2>::likelihoodS1)
      .def("likelihood", &GaussianProcess<2>::likelihood)
      .def("set_data", &GaussianProcess<2>::set_data, py::arg().noconvert(),
           py::arg().noconvert())
      .def("n_parameters", &GaussianProcess<2>::n_parameters)
      .def("set_lengthscale", &GaussianProcess<2>::set_lengthscale)
      .def("set_max_iterations", &GaussianProcess<2>::set_max_iterations)
      .def("set_tolerance", &GaussianProcess<2>::set_tolerance)
      .def("set_trace_iterations", &GaussianProcess<2>::set_trace_iterations)
      .def("set_noise", &GaussianProcess<2>::set_noise);
}
