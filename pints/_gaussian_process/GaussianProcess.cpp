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
      m_stochastic_samples_m(10), m_chebyshev_n(15) {
  set_tolerance(1e-6);
  set_max_iterations(1000);
  for (int i = 0; i < D + 1; ++i) {
    m_gradKs.emplace_back(create_dense_operator(m_particles, m_particles,
                                                GradientKernel_t(m_kernel, i)));
  }
  initialise_chebyshev(m_chebyshev_n);
}

template <unsigned int D> void GaussianProcess<D>::initialise() {
  // normalise kernel
  // const double max_eigenvalue = calculate_eigenvalue();
  // std::cout << "max eigenvalue is " << max_eigenvalue << std::endl;
  // m_kernel.set_sigma(m_kernel.get_sigma() / std::sqrt(max_eigenvalue));

  // create operators
  m_K = create_dense_operator(m_particles, m_particles,
                              Kernel_t(m_kernel, std::pow(m_lambda, 2)));

  for (int i = 0; i < D + 1; ++i) {
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
  //std::cout << "error = "<<(m_K*m_invKy-y).norm()<<std::endl;
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

  const double result = -0.5 * calculate_log_det(m_K, m_stochastic_samples_m) -
                        0.5 * y.dot(m_invKy);
  return result;
}

template <unsigned int D> double GaussianProcess<D>::likelihood_exact() {
  if (m_particles.size() == 0) {
    return 0;
  }
  if (m_uninitialised) {
    initialise();
  }

  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());

  const double result =
      -0.5 * calculate_log_det_exact(m_K) - 0.5 * y.dot(m_invKy);
  return result;
}

template <unsigned int D>
typename GaussianProcess<D>::likelihoodS1_return_t
GaussianProcess<D>::likelihoodS1() {
  likelihoodS1_return_t gradient_and_likelihood = likelihoodS1_return_t::Zero();

  if (m_particles.size() == 0) {
    return gradient_and_likelihood;
  }
  if (m_uninitialised) {
    initialise();
  }

  // trace term
  auto result = calculate_log_det_grad(m_K, m_gradKs, m_stochastic_samples_m);
  for (int i = 0; i < result.size(); ++i) {
    gradient_and_likelihood[i] = -0.5 * result[i];
  }

  // second term
  for (int i = 0; i <= D; ++i) {
    gradient_and_likelihood[i] += 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient_and_likelihood[D + 1] += m_lambda * m_invKy.dot(m_invKy);

  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());

  gradient_and_likelihood[D + 2] += -0.5 * y.dot(m_invKy);

  return gradient_and_likelihood;
}

template <unsigned int D>
typename GaussianProcess<D>::likelihoodS1_return_t
GaussianProcess<D>::likelihoodS1_exact() {
  likelihoodS1_return_t gradient_and_likelihood = likelihoodS1_return_t::Zero();

  if (m_particles.size() == 0) {
    return gradient_and_likelihood;
  }
  if (m_uninitialised) {
    initialise();
  }

  // trace term
  auto result = calculate_log_det_grad_exact(m_K, m_gradKs);
  for (int i = 0; i < result.size(); ++i) {
    gradient_and_likelihood[i] = -0.5 * result[i];
  }

  // second term
  for (int i = 0; i <= D; ++i) {
    gradient_and_likelihood[i] += 0.5 * m_invKy.dot(m_gradKs[i] * m_invKy);
  }
  gradient_and_likelihood[D + 1] += m_lambda * m_invKy.dot(m_invKy);

  Eigen::Map<Eigen::VectorXd> y(get<function>(m_particles).data(),
                                m_particles.size());

  gradient_and_likelihood[D + 2] += -0.5 * y.dot(m_invKy);

  return gradient_and_likelihood;
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
void GaussianProcess<D>::set_parameters(vector_parameter_t parameters) {
  for (size_t j = 0; j < D; ++j) {
    m_lengthscales[j] = parameters[j];
  }
  m_kernel.set_lengthscale(m_lengthscales);
  m_kernel.set_sigma(parameters[D]);
  m_lambda = parameters[D+1];
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
void GaussianProcess<D>::initialise_chebyshev(const int n) {

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

  Eigen::MatrixXd M(B.rows(), B.cols());
  B.assemble(M);
  Eigen::VectorXd eigs = M.eigenvalues().cwiseAbs();
  std::array<double, 2> minmax_exact;
  minmax_exact[1] = std::numeric_limits<double>::min();
  minmax_exact[0] = std::numeric_limits<double>::max();
  for (int i = 0; i < eigs.size(); ++i) {
    if (eigs[i] > minmax_exact[1]) {
      minmax_exact[1] = eigs[i];
    }
    if (eigs[i] < minmax_exact[0]) {
      minmax_exact[0] = eigs[i];
    }
  }
  std::cout << "eigenvalue approx range: " << minmax[0] << "-" << minmax[1]
            << std::endl;
  std::cout << "eigenvalue exact  range: " << minmax_exact[0] << "-"
            << minmax_exact[1] << std::endl;
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

template <unsigned int D>
double GaussianProcess<D>::calculate_log_det(const Operator_t &B, const int m) {

  auto minmax = eigenvalue_range(B);
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

  double gamma = 0;
  for (int i = 0; i < m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(m_particles.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    Eigen::VectorXd u = chebyshev_coefficients[0] * v;
    if (n > 1) {
      Eigen::VectorXd w0 = v;
      // Av = (I - scale*B)v = v - scale*B*v
      Eigen::VectorXd w1 = v - scale * (B * v);
      u += chebyshev_coefficients[1] * w1;
      Eigen::VectorXd w2(m_particles.size());
      for (int j = 2; j < n + 1; ++j) {
        //  2*A*w1 - w0 = 2*(I - scale*B)*w1 - w0 = 2*(w1 - scale*B*w1) - w0
        w2 = 2 * (w1 - scale * (B * w1)) - w0;
        u += chebyshev_coefficients[j] * w2;
        w0 = w1;
        w1 = w2;
      }
    }
    gamma += v.dot(u) / m;
  }

  return gamma - m_particles.size() * std::log(scale);
}

template <unsigned int D>
double GaussianProcess<D>::calculate_log_det_exact(const Operator_t &B) {
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

template <unsigned int D>
std::vector<double> GaussianProcess<D>::calculate_log_det_grad_exact(
    const Operator_t &B, const std::vector<GradientOperator_t> &gradB) {

  // put D+2 gradients in first, then normal log_det term in last
  std::vector<double> result(gradB.size() + 2);

  Eigen::MatrixXd M(B.rows(), B.cols());
  B.assemble(M);
  Eigen::LLT<Eigen::MatrixXd> chol(M);

  for (int i = 0; i < gradB.size(); ++i) {
    Eigen::MatrixXd gradM(gradB[i].rows(), gradB[i].cols());
    gradB[i].assemble(gradM);

    result[i] = chol.solve(gradM).trace();
  }
  result[gradB.size()] =
      chol.solve(2 * m_lambda * Eigen::MatrixXd::Identity(B.rows(), B.cols()))
          .trace();

  /*
  double ld = 0;
  for (size_t i = 0; i < M.rows(); ++i) {
    ld += 1.0 / U(i, i);
  }
  ld /= 2;
  result[gradB.size()] = ld;
  */

  double ld = 0;
  auto &U = chol.matrixL();
  for (size_t i = 0; i < M.rows(); ++i) {
    ld += std::log(U(i, i));
  }
  ld *= 2;
  result[gradB.size() + 1] = ld;
  return result;
}

template <unsigned int D>
std::vector<double> GaussianProcess<D>::calculate_log_det_grad(
    const Operator_t &B, const std::vector<GradientOperator_t> &gradB,
    const int m) {

  std::vector<double> result(gradB.size() + 2, 0);

  for (int i = 0; i < m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(m_particles.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    for (int j = 0; j < gradB.size(); ++j) {
      result[j] += v.dot(m_solver.solve(gradB[j] * v));
    }
    result[gradB.size()] += v.dot(m_solver.solve((2 * m_lambda) * v));
  }

  for (int i = 0; i < result.size() - 1; ++i) {
    result[i] /= m;
  }

  // result.back() = calculate_log_det(B, m);

  return result;
}

/*
template <unsigned int D>
std::vector<double> GaussianProcess<D>::calculate_log_det_grad(
    const Operator_t &B, const std::vector<GradientOperator_t> &gradB,
    const int m) {

  std::vector<double> result(gradB.size() + 2, 0);

  auto minmax = eigenvalue_range(B);
  minmax[0] = 0;

  const double scale = 1.0 / (minmax[1] + minmax[0]);
  const double delta = minmax[0] * scale;

  // specify function to interpolate
  auto f = [&](const double x) {
    //return std::log(1 - ((1 - 2 * delta) * x + 1) / 2);
    return std::log(1 - x);
  };

  auto chebyshev_coefficients = calculate_chebyshev_coefficients(
      m_chebyshev_points, m_chebyshev_polynomials, f);

  const int n = chebyshev_coefficients.size();
  for (int i = 0; i < m; ++i) {
    // generate Rademacher random vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(m_particles.size());
    std::transform(v.data(), v.data() + v.size(), v.data(),
                   [](const double i) { return i > 0 ? 1.0 : -1.0; });
    Eigen::VectorXd u = chebyshev_coefficients[0] * v;

    std::vector<Eigen::VectorXd> gradu(gradB.size() + 1);
    for (int i = 0; i < gradB.size() + 1; ++i) {
      gradu[i] = u;
    }
    if (n > 1) {
      Eigen::VectorXd w0 = v;
      Eigen::VectorXd w1 = v - scale * (B * v);
      u += chebyshev_coefficients[1] * w1;
      Eigen::VectorXd w2(m_particles.size());

      std::vector<Eigen::VectorXd> gradw0(gradB.size() + 1);
      std::vector<Eigen::VectorXd> gradw1(gradB.size() + 1);
      std::vector<Eigen::VectorXd> gradw2(gradB.size() + 1);
      for (int i = 0; i < gradB.size() + 1; ++i) {
        gradw0[i].setZero(m_particles.size());
        if (i == gradB.size()) {
          // final gradB is 2 * m_lambda * I
          gradw1[i] = -2 * scale * m_lambda * v;
        } else {
          gradw1[i] = -scale * (gradB[i] * v);
        }
        gradw2[i].resize(m_particles.size());
        gradu[i] += chebyshev_coefficients[1] * gradw1[i];
      }
      for (int j = 2; j < n; ++j) {
        w2 = 2 * (w1 - scale * (B * w1)) - w0;
        u += chebyshev_coefficients[j] * w2;
        w0 = w1;
        w1 = w2;

        for (int i = 0; i < gradB.size() + 1; ++i) {
          if (i == gradB.size()) {
            // final gradB is 2 * m_lambda * I
            gradw2[i] = 2 * (-2 * scale * m_lambda * w1 + w1 -
                             2 * scale * m_lambda * w1) -
                        w0;
          } else {
            gradw2[i] =
                2 * (-scale * (gradB[i] * w1) + w1 - scale * (B * w1)) - w0;
          }
          gradu[i] += chebyshev_coefficients[j] * gradw2[i];
          gradw0[i] = gradw1[i];
          gradw1[i] = gradw2[i];
        }
      }
    }
    for (int i = 0; i < gradB.size() + 1; ++i) {
      result[i] += v.dot(gradu[i]) / m;
    }
    result[gradB.size() + 1] += v.dot(u) / m;
  }
  // for (int i = 0; i < gradB.size()+1; ++i) {
  //  result[i] += m_particles.size() * std::log(minmax[1] + minmax[0]);
  //}
  result[gradB.size() + 1] -=
      m_particles.size() * std::log(scale);
  return result;
}
*/

template <unsigned int D>
double GaussianProcess<D>::predict(const_vector_D_t argx) {

  const double_d x = argx;

  if (m_particles.size() == 0) {
    return 0;
  }
  if (m_uninitialised) {
    initialise();
  }

  double sum = 0;
  for (int i = 0; i < m_particles.size(); ++i) {
    sum += m_kernel(x, get<position>(m_particles)[i]) * m_invKy(i);
  }
  return sum;
}
template <unsigned int D>
Eigen::Vector2d GaussianProcess<D>::predict_var(const_vector_D_t argx) {
  Eigen::Vector2d mean_var = Eigen::Vector2d::Zero();
  const double_d x = argx;

  if (m_particles.size() == 0) {
    return mean_var;
  }
  if (m_uninitialised) {
    initialise();
  }

  Eigen::VectorXd kstar(m_particles.size());
  for (int i = 0; i < m_particles.size(); ++i) {
    kstar[i] = m_kernel(x, get<position>(m_particles)[i]);
  }

  mean_var[0] = kstar.dot(m_invKy);

  Eigen::VectorXd invKstar = m_solver.solve(kstar);
  if (m_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "invKstar solver failed to converge to set tolerance");
  }
  mean_var[1] = m_kernel(x, x) + std::pow(m_lambda,2) - kstar.dot(invKstar);
  return mean_var;
}

} // namespace Aboria

namespace py = pybind11;
using namespace Aboria;

PYBIND11_MODULE(gaussian_process, m) {

#define ADD_DIMENSION(D)                                                       \
  py::class_<GaussianProcess<D>>(m, "GaussianProcess" #D)                      \
      .def(py::init<>())                                                       \
      .def("likelihoodS1", &GaussianProcess<D>::likelihoodS1)                  \
      .def("likelihood", &GaussianProcess<D>::likelihood)                      \
      .def("likelihoodS1_exact", &GaussianProcess<D>::likelihoodS1_exact)      \
      .def("likelihood_exact", &GaussianProcess<D>::likelihood_exact)          \
      .def("predict", &GaussianProcess<D>::predict)                            \
      .def("predict_var", &GaussianProcess<D>::predict_var)                    \
      .def("set_data", &GaussianProcess<D>::set_data, py::arg().noconvert(),   \
           py::arg().noconvert())                                              \
      .def("n_parameters", &GaussianProcess<D>::n_parameters)                  \
      .def("set_parameters", &GaussianProcess<D>::set_parameters)              \
      .def("set_max_iterations", &GaussianProcess<D>::set_max_iterations)      \
      .def("set_tolerance", &GaussianProcess<D>::set_tolerance)                \
      .def("set_chebyshev_n", &GaussianProcess<D>::set_chebyshev_n)            \
      .def("set_stochastic_samples",                                           \
           &GaussianProcess<D>::set_stochastic_samples);

  ADD_DIMENSION(1)
  ADD_DIMENSION(2)
}
