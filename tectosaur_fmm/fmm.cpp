<% 
from tectosaur_fmm.cfg import lib_cfg
lib_cfg(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "include/pybind11_nparray.hpp"

#include "fmm_impl.hpp"
#include "octree.hpp"

namespace py = pybind11;


int main(int,char**);

template <size_t dim>
void wrap_dim(py::module& m) {
    std::string dim_str = std::to_string(dim);

    py::class_<Cube<dim>>(m, "Cube")
        .def_readonly("center", &Cube<dim>::center)
        .def_readonly("width", &Cube<dim>::width);

    m.def("in_box", &in_box<dim>);

    py::class_<OctreeNode<dim>>(m, "OctreeNode")
        .def_readonly("start", &OctreeNode<dim>::start)
        .def_readonly("end", &OctreeNode<dim>::end)
        .def_readonly("bounds", &OctreeNode<dim>::bounds)
        .def_readonly("is_leaf", &OctreeNode<dim>::is_leaf)
        .def_readonly("idx", &OctreeNode<dim>::idx)
        .def_readonly("height", &OctreeNode<dim>::height)
        .def_readonly("depth", &OctreeNode<dim>::depth)
        .def_readonly("children", &OctreeNode<dim>::children);

    py::class_<Octree<dim>>(m, "Octree")
        .def("__init__",
        [] (Octree<dim>& kd, NPArrayD np_pts, NPArrayD np_normals, size_t n_per_cell) {
            check_shape<dim>(np_pts);
            check_shape<dim>(np_normals);
            new (&kd) Octree<dim>(
                reinterpret_cast<std::array<double,dim>*>(np_pts.request().ptr),
                reinterpret_cast<std::array<double,dim>*>(np_normals.request().ptr),
                np_pts.request().shape[0], n_per_cell
            );
        })
        .def("root", &Octree<dim>::root)
        .def_readonly("nodes", &Octree<dim>::nodes)
        .def_readonly("pts", &Octree<dim>::pts)
        .def_readonly("normals", &Octree<dim>::normals)
        .def_readonly("orig_idxs", &Octree<dim>::orig_idxs)
        .def_readonly("max_height", &Octree<dim>::max_height);

    py::class_<FMMConfig<dim>>(m, "FMMConfig")
        .def("__init__", 
            [] (FMMConfig<dim>& cfg, double equiv_r,
                double check_r, size_t order, std::string k_name,
                NPArrayD params) 
            {
                new (&cfg) FMMConfig<dim>{
                    equiv_r, check_r, order, get_by_name<dim>(k_name),
                    get_vector<double>(params)                                    
                };
            }
        )
        .def_readonly("inner_r", &FMMConfig<dim>::inner_r)
        .def_readonly("outer_r", &FMMConfig<dim>::outer_r)
        .def_readonly("order", &FMMConfig<dim>::order)
        .def_readonly("params", &FMMConfig<dim>::params)
        .def("kernel_name", &FMMConfig<dim>::kernel_name)
        .def("tensor_dim", &FMMConfig<dim>::tensor_dim);

    py::class_<FMMMat<dim>>(m, "FMMMat")
        .def_readonly("obs_tree", &FMMMat<dim>::obs_tree)
        .def_readonly("src_tree", &FMMMat<dim>::src_tree)
        .def_readonly("surf", &FMMMat<dim>::surf)
        .def_readonly("p2p", &FMMMat<dim>::p2p)
        .def_readonly("m2p", &FMMMat<dim>::m2p)
        .def_readonly("cfg", &FMMMat<dim>::cfg)
        .def_readonly("translation_surface_order", &FMMMat<dim>::translation_surface_order)
        .def_readonly("uc2e", &FMMMat<dim>::uc2e)
        .def_property_readonly("tensor_dim", &FMMMat<dim>::tensor_dim)
        .def("p2p_eval", [] (FMMMat<dim>& m, NPArrayD v) {
            auto* ptr = reinterpret_cast<double*>(v.request().ptr);
            return array_from_vector(m.p2p_eval(ptr));
        })
        .def("p2m_eval", [] (FMMMat<dim>& m, NPArrayD v) {
            auto* ptr = reinterpret_cast<double*>(v.request().ptr);
            return array_from_vector(m.p2m_eval(ptr));
        })
        .def("m2p_eval", [] (FMMMat<dim>& m, NPArrayD multipoles) {
            auto* ptr = reinterpret_cast<double*>(multipoles.request().ptr);
            return array_from_vector(m.m2p_eval(ptr));
        });

    m.def("fmmmmmmm", &fmmmmmmm<dim>);
}


PYBIND11_PLUGIN(fmm) {
    py::module m("fmm");
    auto two = m.def_submodule("two");
    auto three = m.def_submodule("three");

    wrap_dim<2>(two);
    wrap_dim<3>(three);

    py::class_<BlockSparseMat>(m, "BlockSparseMat")
        .def("get_nnz", &BlockSparseMat::get_nnz)
        .def("matvec", [] (BlockSparseMat& s, NPArrayD v, size_t n_rows) {
            auto out = s.matvec(reinterpret_cast<double*>(v.request().ptr), n_rows);
            return array_from_vector(out);
        });

    py::class_<MatrixFreeOp>(m, "MatrixFreeOp")
        .def_readonly("obs_n_start", &MatrixFreeOp::obs_n_start)
        .def_readonly("obs_n_end", &MatrixFreeOp::obs_n_end)
        .def_readonly("obs_n_idx", &MatrixFreeOp::obs_n_idx)
        .def_readonly("src_n_start", &MatrixFreeOp::src_n_start)
        .def_readonly("src_n_end", &MatrixFreeOp::src_n_end)
        .def_readonly("src_n_idx", &MatrixFreeOp::src_n_idx);


    m.def("direct_eval", [](std::string k_name, NPArrayD obs_pts, NPArrayD obs_ns,
                            NPArrayD src_pts, NPArrayD src_ns, NPArrayD params) {
        check_shape<3>(obs_pts);
        check_shape<3>(obs_ns);
        check_shape<3>(src_pts);
        check_shape<3>(src_ns);
        auto K = get_by_name<3>(k_name);
        std::vector<double> out(K.tensor_dim * K.tensor_dim *
                                obs_pts.request().shape[0] *
                                src_pts.request().shape[0]);
        K.f({as_ptr<std::array<double,3>>(obs_pts), as_ptr<std::array<double,3>>(obs_ns),
           as_ptr<std::array<double,3>>(src_pts), as_ptr<std::array<double,3>>(src_ns),
           obs_pts.request().shape[0], src_pts.request().shape[0],
           as_ptr<double>(params)},
          out.data());
        return array_from_vector(out);
    });

    m.def("mf_direct_eval", [](std::string k_name, NPArrayD obs_pts, NPArrayD obs_ns,
                            NPArrayD src_pts, NPArrayD src_ns, NPArrayD params, NPArrayD input) {
        check_shape<3>(obs_pts);
        check_shape<3>(obs_ns);
        check_shape<3>(src_pts);
        check_shape<3>(src_ns);
        auto K = get_by_name<3>(k_name);
        std::vector<double> out(K.tensor_dim * obs_pts.request().shape[0]);
        K.f_mf({as_ptr<std::array<double,3>>(obs_pts), as_ptr<std::array<double,3>>(obs_ns),
           as_ptr<std::array<double,3>>(src_pts), as_ptr<std::array<double,3>>(src_ns),
           obs_pts.request().shape[0], src_pts.request().shape[0],
           as_ptr<double>(params)},
          out.data(), as_ptr<double>(input));
        return array_from_vector(out);
    });
    return m.ptr();
}
