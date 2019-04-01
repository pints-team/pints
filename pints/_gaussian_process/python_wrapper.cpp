#include "GaussianProcessMatrixFree.hpp"
#include "GaussianProcessDenseMatrix.hpp"

namespace py = pybind11;
using namespace Aboria;

PYBIND11_MODULE(gaussian_process, m) {

#define ADD_DIMENSION(D)                                                       \
  py::class_<GaussianProcessMatrixFree<D>>(m, "GaussianProcessMatrixFree" #D)  \
      .def(py::init<>())                                                       \
      .def("grad_likelihood", &GaussianProcessMatrixFree<D>::grad_likelihood)  \
      .def("likelihood", &GaussianProcessMatrixFree<D>::likelihood)            \
      .def("predict", &GaussianProcessMatrixFree<D>::predict)                  \
      .def("predict_var", &GaussianProcessMatrixFree<D>::predict_var)          \
      .def("set_data", &GaussianProcessMatrixFree<D>::set_data,                \
           py::arg().noconvert(), py::arg().noconvert())                       \
      .def("n_parameters", &GaussianProcessMatrixFree<D>::n_parameters)        \
      .def("set_parameters", &GaussianProcessMatrixFree<D>::set_parameters)    \
      .def("set_max_iterations",                                               \
           &GaussianProcessMatrixFree<D>::set_max_iterations)                  \
      .def("set_tolerance", &GaussianProcessMatrixFree<D>::set_tolerance)      \
      .def("set_chebyshev_n", &GaussianProcessMatrixFree<D>::set_chebyshev_n)  \
      .def("set_stochastic_samples",                                           \
           &GaussianProcessMatrixFree<D>::set_stochastic_samples);             \
                                                                               \
  py::class_<GaussianProcessDenseMatrix<D>>(m,                                 \
                                            "GaussianProcessDenseMatrix" #D)   \
      .def(py::init<>())                                                       \
      .def("grad_likelihood", &GaussianProcessDenseMatrix<D>::grad_likelihood) \
      .def("likelihood", &GaussianProcessDenseMatrix<D>::likelihood)           \
      .def("predict", &GaussianProcessDenseMatrix<D>::predict)                 \
      .def("predict_var", &GaussianProcessDenseMatrix<D>::predict_var)         \
      .def("set_data", &GaussianProcessDenseMatrix<D>::set_data,               \
           py::arg().noconvert(), py::arg().noconvert())                       \
      .def("n_parameters", &GaussianProcessDenseMatrix<D>::n_parameters)       \
      .def("set_parameters", &GaussianProcessDenseMatrix<D>::set_parameters)   \
      .def("set_max_iterations",                                               \
           &GaussianProcessDenseMatrix<D>::set_max_iterations)                 \
      .def("set_tolerance", &GaussianProcessDenseMatrix<D>::set_tolerance)     \
      .def("set_chebyshev_n", &GaussianProcessDenseMatrix<D>::set_chebyshev_n) \
      .def("set_stochastic_samples",                                           \
           &GaussianProcessDenseMatrix<D>::set_stochastic_samples);

  ADD_DIMENSION(1)
  ADD_DIMENSION(2)
}
