<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-g', '-Wall', '-Werror', '-fopenmp'])
cfg['include_dirs'] = cfg.get('include_dirs',[]) + ['../effemmemm']
cfg['sources'] = cfg.get('sources',[]) + ['test_blas.cpp']
cfg['sources'] += ['../effemmemm/blas_wrapper.cpp']
cfg['dependencies'] = ['../effemmemm/blas_wrapper.hpp']

import numpy as np
blas = np.__config__.blas_opt_info
cfg['library_dirs'] = blas['library_dirs']
cfg['libraries'] = blas['libraries']
%>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_PLUGIN(test_main) {
    py::module m("test_main");

    m.def("run_tests", [] (std::vector<std::string> str_args) { 
        char** argv = new char*[str_args.size()];
        for (size_t i = 0; i < str_args.size(); i++) {
            argv[i] = const_cast<char*>(str_args[i].c_str());
        }
        main(str_args.size(), argv); 
        delete[] argv;
    });

    return m.ptr();
}
