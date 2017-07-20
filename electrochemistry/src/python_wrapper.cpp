#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "utilities.hpp"
#include "seq_electron_transfer3_explicit.hpp"
#include "e_implicit_exponential_mesh.hpp"

namespace py = boost::python;
using namespace pints;

template <class C>
struct pickle_suite: public py::pickle_suite { BOOST_STATIC_ASSERT(sizeof(C)==0); };

template <typename  T>
struct pickle_suite< std::vector<T> >: public py::pickle_suite
{
    static py::tuple getinitargs(const std::vector<T>& o)
    {
        return py::make_tuple();
    }

    static py::tuple getstate(py::object obj)
    {
        const std::vector<T>& o = py::extract<const std::vector<T>&>(obj)();

        return py::make_tuple(py::list(o));
    }

    static void setstate(py::object obj, py::tuple state)
    {
        std::vector<T>& o = py::extract<std::vector<T>&>(obj)();

        py::stl_input_iterator<typename std::vector<T>::value_type> begin(state[0]), end;
        o.insert(o.begin(),begin,end);
    }
};


template <typename K, typename T>
struct pickle_suite< std::map<K,T> >: public py::pickle_suite
{
    static py::tuple getinitargs(const std::map<K,T>& o)
    {
        return py::make_tuple();
    }

    static py::tuple getstate(py::object obj)
    {
        const std::map<K,T>& o = py::extract<const std::map<K,T>&>(obj)();

	    boost::python::dict dict;
        typename std::map<K, T>::const_iterator iter;
        for (iter = o.begin(); iter != o.end(); ++iter) {
            dict[iter->first] = iter->second;
        }
        

        return py::make_tuple(dict);
    }

    static void setstate(py::object obj, py::tuple state)
    {
        std::map<K,T>& o = py::extract<std::map<K,T>&>(obj)();

        py::dict test = py::extract<py::dict>(state[0]);
        const size_t n = py::len(test);
        for (int i = 0; i < n; ++i) {
            K key = py::extract<K>(test.keys()[i]);
            T value = py::extract<T>(test.values()[i]);
            o[key] = value;
        }
    }
};

BOOST_PYTHON_MODULE(pints_cpp)
{

        py::class_<vector>("pints_vector")
            .def(py::vector_indexing_suite<vector>())
            .def_pickle(pickle_suite<vector>());
            ;

        py::class_<map>("pints_map")
            .def(py::map_indexing_suite<map>())
            .def_pickle(pickle_suite<map>());
            ;

        py::def("e_implicit_exponential_mesh", e_implicit_exponential_mesh, (py::arg("params"), py::arg("Itot"), py::arg("t")), "Solves E problem using implicit thomas method, exponentially expanding mesh");
        py::def("seq_electron_transfer3_explicit", seq_electron_transfer3_explicit, (py::arg("params"), py::arg("Itot"), py::arg("t")), "placeholder");
        }
