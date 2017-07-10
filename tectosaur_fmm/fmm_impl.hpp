#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include "fmm_kernels.hpp"
#include "octree.hpp"

template <size_t dim>
std::vector<std::array<double,dim>> inscribe_surf(const Cube<dim>& b, double scaling,
                                const std::vector<std::array<double,dim>>& fmm_surf) {
    std::vector<std::array<double,dim>> out(fmm_surf.size());
    for (size_t i = 0; i < fmm_surf.size(); i++) {
        for (size_t d = 0; d < dim; d++) {
            out[i][d] = fmm_surf[i][d] * b.R() * scaling + b.center[d];
        }
    }
    return out;
}

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

struct Block {
    size_t row_start;
    size_t col_start;
    int n_rows;
    int n_cols;
    size_t data_start;
};

struct BlockSparseMat {
    std::vector<Block> blocks;
    std::vector<double> vals;

    std::vector<double> matvec(double* vec, size_t out_size);
    size_t get_nnz() { return vals.size(); }
};

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

    std::vector<BlockSparseMat> uc2e;

    FMMMat(Octree<dim> obs_tree, Octree<dim> src_tree, FMMConfig<dim> cfg,
        std::vector<std::array<double,dim>> surf);

    std::vector<std::array<double,dim>> get_surf(const OctreeNode<dim>& src_n, double r);
    
    int tensor_dim() const { return cfg.tensor_dim(); }

    void p2m_matvec(double* out, double* in);
    void m2m_matvec(double* out, double* in, int level);
    void p2p_matvec(double* out, double* in);
    void m2p_matvec(double* out, double* in);

    std::vector<double> p2p_eval(double* in);
    std::vector<double> p2m_eval(double* in);
    std::vector<double> m2p_eval(double* multipoles);
};

template <size_t dim>
FMMMat<dim> fmmmmmmm(const Octree<dim>& obs_tree, const Octree<dim>& src_tree,
    const FMMConfig<dim>& cfg);
