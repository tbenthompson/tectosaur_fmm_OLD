#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include "fmm_kernels.hpp"
#include "octree.hpp"
#include "blas_wrapper.hpp"
#include "translation_surf.hpp"

template <size_t dim>
struct FMMConfig {
    // The MAC needs to < (1.0 / (check_r - 1)) so that farfield
    // approximations aren't used when the check surface intersects the
    // target box. How about we just use that as our MAC, since errors
    // further from the check surface should flatten out!
    double inner_r;
    double outer_r;
    size_t order;
    Kernel<dim> kernel;
    std::vector<double> params;

    std::string kernel_name() const { return kernel.name; }
    int tensor_dim() const { return kernel.tensor_dim; }
};

template <size_t dim>
std::vector<double> c2e_solve(std::vector<std::array<double,dim>> surf,
    const Cube<dim>& bounds, double check_r, double equiv_r, const FMMConfig<dim>& cfg) 
{
    auto equiv_surf = inscribe_surf(bounds, equiv_r, surf);
    auto check_surf = inscribe_surf(bounds, check_r, surf);

    auto n_surf = surf.size();
    auto n_rows = n_surf * cfg.tensor_dim();

    std::vector<double> equiv_to_check(n_rows * n_rows);
    cfg.kernel.f(
        {
            check_surf.data(), surf.data(), 
            equiv_surf.data(), surf.data(),
            n_surf, n_surf,
            cfg.params.data()
        },
        equiv_to_check.data());

    auto pinv = qr_pseudoinverse(equiv_to_check.data(), n_rows);

    return pinv;
}

struct MatrixFreeOp {
    std::vector<int> obs_n_start;
    std::vector<int> obs_n_end;
    std::vector<int> obs_n_idx;
    std::vector<int> src_n_start;
    std::vector<int> src_n_end;
    std::vector<int> src_n_idx;

    template <size_t dim>
    void insert(const OctreeNode<dim>& obs_n, const OctreeNode<dim>& src_n) {
        obs_n_start.push_back(obs_n.start);
        obs_n_end.push_back(obs_n.end);
        obs_n_idx.push_back(obs_n.idx);
        src_n_start.push_back(src_n.start);
        src_n_end.push_back(src_n.end);
        src_n_idx.push_back(src_n.idx);
    }
};

template <size_t dim>
struct FMMMat {
    Octree<dim> obs_tree;
    Octree<dim> src_tree;
    FMMConfig<dim> cfg;
    std::vector<std::array<double,dim>> surf;
    int translation_surface_order;

    MatrixFreeOp p2p;
    MatrixFreeOp p2m;
    MatrixFreeOp m2p;
    std::vector<MatrixFreeOp> m2m;

    std::vector<double> uc2e_ops;
    std::vector<MatrixFreeOp> uc2e;

    FMMMat(Octree<dim> obs_tree, Octree<dim> src_tree, FMMConfig<dim> cfg,
        std::vector<std::array<double,dim>> surf);

    int tensor_dim() const { return cfg.tensor_dim(); }

    void p2m_matvec(double* out, double* in);
    void m2m_matvec(double* out, double* in, int level);
    void p2p_matvec(double* out, double* in);
    void m2p_matvec(double* out, double* in);
    void uc2e_matvec(double* out, double* in, int level);

    std::vector<double> m2m_eval(double* m_check);
    std::vector<double> m2p_eval(double* multipoles);
};

template <size_t dim>
FMMMat<dim> fmmmmmmm(const Octree<dim>& obs_tree, const Octree<dim>& src_tree,
    const FMMConfig<dim>& cfg);
