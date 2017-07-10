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
#include "kdtree.hpp"

namespace py = pybind11;


int main(int,char**);

template <size_t dim>
void wrap_octree(py::module& m) {
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
}


PYBIND11_PLUGIN(fmm) {
    py::module m("fmm");
    auto two = m.def_submodule("two");
    auto three = m.def_submodule("three");

    wrap_octree<2>(two);
    wrap_octree<3>(three);

    py::class_<Sphere>(m, "Sphere")
        .def_readonly("r", &Sphere::r)
        .def_readonly("center", &Sphere::center);

    py::class_<KDNode>(m, "KDNode")
        .def_readonly("start", &KDNode::start)
        .def_readonly("end", &KDNode::end)
        .def_readonly("bounds", &KDNode::bounds)
        .def_readonly("is_leaf", &KDNode::is_leaf)
        .def_readonly("idx", &KDNode::idx)
        .def_readonly("height", &KDNode::height)
        .def_readonly("depth", &KDNode::depth)
        .def_readonly("children", &KDNode::children);

    py::class_<KDTree>(m, "KDTree")
        .def("__init__", 
        [] (KDTree& kd, NPArrayD np_pts, NPArrayD np_normals, size_t n_per_cell) {
            check_shape<3>(np_pts);
            check_shape<3>(np_normals);
            new (&kd) KDTree(
                reinterpret_cast<Vec3*>(np_pts.request().ptr),
                reinterpret_cast<Vec3*>(np_normals.request().ptr),
                np_pts.request().shape[0], n_per_cell
            );
        })
        .def_readonly("nodes", &KDTree::nodes)
        .def_readonly("pts", &KDTree::pts)
        .def_readonly("normals", &KDTree::normals)
        .def_readonly("orig_idxs", &KDTree::orig_idxs)
        .def_readonly("max_height", &KDTree::max_height);

    py::class_<BlockSparseMat>(m, "BlockSparseMat")
        .def("get_nnz", &BlockSparseMat::get_nnz)
        .def("matvec", [] (BlockSparseMat& s, NPArrayD v, size_t n_rows) {
            auto out = s.matvec(reinterpret_cast<double*>(v.request().ptr), n_rows);
            return array_from_vector(out);
        });

    py::class_<FMMConfig>(m, "FMMConfig")
        .def("__init__", 
            [] (FMMConfig& cfg, double equiv_r,
                double check_r, size_t order, std::string k_name,
                NPArrayD params) 
            {
                new (&cfg) FMMConfig{
                    equiv_r, check_r, order, get_by_name(k_name),
                    get_vector<double>(params)                                    
                };
            }
        )
        .def_readonly("inner_r", &FMMConfig::inner_r)
        .def_readonly("outer_r", &FMMConfig::outer_r)
        .def_readonly("order", &FMMConfig::order)
        .def_readonly("params", &FMMConfig::params)
        .def("kernel_name", &FMMConfig::kernel_name)
        .def("tensor_dim", &FMMConfig::tensor_dim);

    py::class_<MatrixFreeOp>(m, "MatrixFreeOp")
        .def_readonly("obs_n_start", &MatrixFreeOp::obs_n_start)
        .def_readonly("obs_n_end", &MatrixFreeOp::obs_n_end)
        .def_readonly("obs_n_idx", &MatrixFreeOp::obs_n_idx)
        .def_readonly("src_n_start", &MatrixFreeOp::src_n_start)
        .def_readonly("src_n_end", &MatrixFreeOp::src_n_end)
        .def_readonly("src_n_idx", &MatrixFreeOp::src_n_idx);

    py::class_<FMMMat>(m, "FMMMat")
        .def_readonly("obs_tree", &FMMMat::obs_tree)
        .def_readonly("src_tree", &FMMMat::src_tree)
        .def_readonly("surf", &FMMMat::surf)
        .def_readonly("p2p", &FMMMat::p2p)
        .def_readonly("m2p", &FMMMat::m2p)
        .def_readonly("cfg", &FMMMat::cfg)
        .def_readonly("translation_surface_order", &FMMMat::translation_surface_order)
        .def_readonly("uc2e", &FMMMat::uc2e)
        .def_property_readonly("tensor_dim", &FMMMat::tensor_dim)
        .def("p2p_eval", [] (FMMMat& m, NPArrayD v) {
            auto* ptr = reinterpret_cast<double*>(v.request().ptr);
            return array_from_vector(m.p2p_eval(ptr));
        })
        .def("p2m_eval", [] (FMMMat& m, NPArrayD v) {
            auto* ptr = reinterpret_cast<double*>(v.request().ptr);
            return array_from_vector(m.p2m_eval(ptr));
        })
        .def("m2p_eval", [] (FMMMat& m, NPArrayD multipoles) {
            auto* ptr = reinterpret_cast<double*>(multipoles.request().ptr);
            return array_from_vector(m.m2p_eval(ptr));
        });

    m.def("fmmmmmmm", &fmmmmmmm);

    m.def("direct_eval", [](std::string k_name, NPArrayD obs_pts, NPArrayD obs_ns,
                            NPArrayD src_pts, NPArrayD src_ns, NPArrayD params) {
        check_shape<3>(obs_pts);
        check_shape<3>(obs_ns);
        check_shape<3>(src_pts);
        check_shape<3>(src_ns);
        auto K = get_by_name(k_name);
        std::vector<double> out(K.tensor_dim * K.tensor_dim *
                                obs_pts.request().shape[0] *
                                src_pts.request().shape[0]);
        K.f({as_ptr<Vec3>(obs_pts), as_ptr<Vec3>(obs_ns),
           as_ptr<Vec3>(src_pts), as_ptr<Vec3>(src_ns),
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
        auto K = get_by_name(k_name);
        std::vector<double> out(K.tensor_dim * obs_pts.request().shape[0]);
        K.f_mf({as_ptr<Vec3>(obs_pts), as_ptr<Vec3>(obs_ns),
           as_ptr<Vec3>(src_pts), as_ptr<Vec3>(src_ns),
           obs_pts.request().shape[0], src_pts.request().shape[0],
           as_ptr<double>(params)},
          out.data(), as_ptr<double>(input));
        return array_from_vector(out);
    });
    return m.ptr();
}
