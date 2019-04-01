#pragma once

#include "Aboria.h"
#include <Eigen/Core>

namespace Aboria {

template <unsigned int D> struct matern_kernel {
  using gradient_t = Eigen::Matrix<double, D, 1>;
  using double_d = Vector<double, D>;
  double_d m_lengthscale;
  double m_sigma2;
  double m_sigma;
  static constexpr const char *m_name = "matern";
  matern_kernel(const double sigma, const double_d &lengthscales) {
    set_sigma(sigma);
    set_lengthscale(lengthscales);
  }
  void set_sigma(const double sigma) {
    m_sigma = sigma;
    m_sigma2 = std::pow(sigma, 2);
  }
  double get_sigma() { return m_sigma; }
  void set_lengthscale(const double_d &lengthscale) {
    for (int i = 0; i < D; ++i) {
      m_lengthscale[i] = 1.0 / lengthscale[i];
    }
  }
  double operator()(const Vector<double, D> &a,
                    const Vector<double, D> &b) const {
    const double r = ((b - a) * m_lengthscale).norm();
    return m_sigma2 * (1.0 + std::sqrt(3.0) * r) *
           std::exp(-std::sqrt(3.0) * r);
  }

  double gradient_by_lengthscale(const Vector<double, D> &a,
                                 const Vector<double, D> &b,
                                 const int i) const {
    const double_d dx2 = ((b - a) * m_lengthscale).pow(2);
    const double r = std::sqrt(dx2.sum());
    const double exp_term = std::exp(-std::sqrt(3.0) * r);
    const double factor = 3 * m_sigma2 * exp_term;
    return m_lengthscale[i] * dx2[i] * factor;
  }

  double gradient_by_sigma(const Vector<double, D> &a,
                           const Vector<double, D> &b) const {
    const double_d dx2 = ((b - a) * m_lengthscale).pow(2);
    const double r = std::sqrt(dx2.sum());
    const double exp_term = std::exp(-std::sqrt(3.0) * r);
    return 2 * m_sigma * (1.0 + std::sqrt(3.0) * r) * exp_term;
  }
};

template <typename Particles, typename Kernel> struct self_kernel {
  using raw_const_reference = typename Particles::raw_const_reference;
  using position = typename Particles::position;
  Kernel m_kernel;
  double m_diagonal;

  self_kernel(const Kernel &kernel, const double diagonal)
      : m_kernel(kernel), m_diagonal(diagonal) {}

  template <typename OldKernel> self_kernel(const OldKernel &kernel) {
    m_kernel.m_scale = kernel.m_kernel.m_scale;
    m_kernel.m_sigma = kernel.m_kernel.m_sigma;
    m_diagonal = kernel.m_diagonal;
  }

  double operator()(raw_const_reference a, raw_const_reference b) const {
    double ret = m_kernel(get<position>(a), get<position>(b));
    if (get<id>(a) == get<id>(b)) {
      ret += m_diagonal;
    }
    return ret;
  }
};

template <typename Particles, typename Kernel>
struct gradient_by_lengthscale_kernel {
  using raw_const_reference = typename Particles::raw_const_reference;
  using position = typename Particles::position;
  Kernel m_kernel;
  int m_i;

  gradient_by_lengthscale_kernel(const Kernel &kernel, const int i)
      : m_kernel(kernel), m_i(i) {}

  double operator()(raw_const_reference a, raw_const_reference b) const {
    double ret = m_kernel.gradient_by_lengthscale(get<position>(a),
                                                  get<position>(b), m_i);
    if (get<id>(a) == get<id>(b)) {
      ret += 1e-5;
    }
    return ret;
  }
};

template <typename Particles, typename Kernel> struct gradient_by_sigma_kernel {
  using raw_const_reference = typename Particles::raw_const_reference;
  using position = typename Particles::position;
  Kernel m_kernel;

  gradient_by_sigma_kernel(const Kernel &kernel) : m_kernel(kernel) {}

  double operator()(raw_const_reference a, raw_const_reference b) const {
    double ret = m_kernel.gradient_by_sigma(get<position>(a), get<position>(b));
    if (get<id>(a) == get<id>(b)) {
      ret += 1e-5;
    }
    return ret;
  }
};

} // namespace Aboria
