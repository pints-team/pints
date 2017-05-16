#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "utilities.hpp"
#include "e_implicit_exponential_mesh.hpp"

BOOST_PYTHON_MODULE(hobo_cpp)
{
        using namespace boost::python;
        using namespace hobo;

        class_<vector>("hobo_vector")
            .def(vector_indexing_suite<vector>())
            ;

        class_<map>("hobo_map")
            .def(map_indexing_suite<map>())
            ;

        def("e_implicit_exponential_mesh", e_implicit_exponential_mesh, (arg("params"), arg("Itot"), arg("t")), "Solves E problem using implicit thomas method, exponentially expanding mesh");
        }
