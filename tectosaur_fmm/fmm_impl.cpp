#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "include/timing.hpp"
#include "fmm_impl.hpp"

template <size_t dim>
void p2p(FMMMat<dim>& mat, const OctreeNode<dim>& obs_n, const OctreeNode<dim>& src_n) {
    mat.p2p.insert(obs_n, src_n);
}

template <size_t dim>
void m2p(FMMMat<dim>& mat, const OctreeNode<dim>& obs_n, const OctreeNode<dim>& src_n) {
    mat.m2p.insert(obs_n, src_n);
}

template <size_t dim>
void p2m(FMMMat<dim>& mat, const OctreeNode<dim>& src_n) {
    mat.p2m.insert(src_n, src_n);
}

template <size_t dim>
void m2m(FMMMat<dim>& mat, const OctreeNode<dim>& parent_n, const OctreeNode<dim>& child_n) {
    mat.m2m[parent_n.height].insert(parent_n, child_n);
}

int traverse_touches = 0;
template <size_t dim>
void traverse(FMMMat<dim>& mat, const OctreeNode<dim>& obs_n, const OctreeNode<dim>& src_n) {
    traverse_touches++;
    auto r_src = src_n.bounds.R();
    auto r_obs = obs_n.bounds.R();
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    // If outer_r * r_src + inner_r * r_obs is less than the separation, then
    // the relevant check surfaces for the two interacting cells don't
    // intersect.
    // That means it should be safe to perform approximate interactions. I add
    // a small safety factor just in case!
    double safety_factor = 0.98;
    if (mat.cfg.outer_r * r_src + mat.cfg.inner_r * r_obs < safety_factor * sep) {
        // If there aren't enough src or obs to justify using the approximation,
        // then just do a p2p direct calculation between the nodes.
        bool small_src = src_n.end - src_n.start < mat.surf.size();

        if (small_src) {
            p2p(mat, obs_n, src_n);
        } else {
            m2p(mat, obs_n, src_n);
        }
        return;
    }

    if (src_n.is_leaf && obs_n.is_leaf) {
        p2p(mat, obs_n, src_n);
        return;
    }

    bool split_src = ((r_obs < r_src) && !src_n.is_leaf) || obs_n.is_leaf;
    if (split_src) {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            traverse(mat, obs_n, mat.src_tree.nodes[src_n.children[i]]);
        }
    } else {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            traverse(mat, mat.obs_tree.nodes[obs_n.children[i]], src_n);
        }
    }
}

template <size_t dim>
void c2e(FMMMat<dim>& mat, BlockSparseMat& sub_mat, const OctreeNode<dim>& node,
         double check_r, double equiv_r) {

    mat.uc2e[node.height].insert(node, node);

    // auto& pinv = mat.uc2e_ops[node.depth];
    // auto n_rows = mat.surf.size() * mat.cfg.tensor_dim();
    // sub_mat.blocks.push_back({
    //     node.idx * n_rows, node.idx * n_rows,
    //     int(n_rows), int(n_rows),
    //     sub_mat.vals.size()
    // });
    // sub_mat.vals.insert(sub_mat.vals.end(), pinv.begin(), pinv.end());
}

int up_collect_touches = 0;
template <size_t dim>
void up_collect(FMMMat<dim>& mat, const OctreeNode<dim>& src_n) {
    up_collect_touches++;
    c2e(mat, mat.uc2e_old[src_n.height], src_n, mat.cfg.outer_r, mat.cfg.inner_r);
    if (src_n.is_leaf) {
        p2m(mat, src_n);
    } else {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            auto child = mat.src_tree.nodes[src_n.children[i]];
            up_collect(mat, child);
            m2m(mat, src_n, child);
        }
    }
}

template <size_t dim>
FMMMat<dim>::FMMMat(Octree<dim> obs_tree, Octree<dim> src_tree, FMMConfig<dim> cfg,
        std::vector<std::array<double,dim>> surf):
    obs_tree(obs_tree),
    src_tree(src_tree),
    cfg(cfg),
    surf(surf),
    translation_surface_order(surf.size())
{}

template <size_t dim>
void interact_pts(const FMMConfig<dim>& cfg, double* out, double* in,
    const std::array<double,dim>* obs_pts, const std::array<double,dim>* obs_ns,
    size_t n_obs, size_t obs_pt_start,
    const std::array<double,dim>* src_pts, const std::array<double,dim>* src_ns,
    size_t n_src, size_t src_pt_start) 
{
    if (n_obs == 0 || n_src == 0) {
        return;
    }

    double* out_val_start = &out[cfg.tensor_dim() * obs_pt_start];
    double* in_val_start = &in[cfg.tensor_dim() * src_pt_start];
    cfg.kernel.f_mf(
        NBodyProblem<dim>{obs_pts, obs_ns, src_pts, src_ns, n_obs, n_src, cfg.params.data()},
        out_val_start, in_val_start
    );
}


template <size_t dim>
void FMMMat<dim>::p2p_matvec(double* out, double* in) {
    for (size_t i = 0; i < p2p.obs_n_idx.size(); i++) {
        auto obs_n = obs_tree.nodes[p2p.obs_n_idx[i]];
        auto src_n = src_tree.nodes[p2p.src_n_idx[i]];
        interact_pts(
            cfg, out, in,
            &obs_tree.pts[obs_n.start], &obs_tree.normals[obs_n.start],
            obs_n.end - obs_n.start, obs_n.start,
            &src_tree.pts[src_n.start], &src_tree.normals[src_n.start],
            src_n.end - src_n.start, src_n.start
        );
    }
}

template <size_t dim>
void FMMMat<dim>::p2m_matvec(double* out, double *in) {
    for (size_t i = 0; i < p2m.obs_n_idx.size(); i++) {
        auto src_n = src_tree.nodes[p2m.src_n_idx[i]];
        auto check = inscribe_surf(src_n.bounds, cfg.outer_r, surf);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), src_n.idx * surf.size(),
            &src_tree.pts[src_n.start], &src_tree.normals[src_n.start],
            src_n.end - src_n.start, src_n.start
        );
    }
}

template <size_t dim>
void FMMMat<dim>::m2m_matvec(double* out, double *in, int level) {
    for (size_t i = 0; i < m2m[level].obs_n_idx.size(); i++) {
        auto parent_n = src_tree.nodes[m2m[level].obs_n_idx[i]];
        auto child_n = src_tree.nodes[m2m[level].src_n_idx[i]];
        auto check = inscribe_surf(parent_n.bounds, cfg.outer_r, surf);
        auto equiv = inscribe_surf(child_n.bounds, cfg.inner_r, surf);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), parent_n.idx * surf.size(),
            equiv.data(), surf.data(), 
            surf.size(), child_n.idx * surf.size()
        );
    }
}

template <size_t dim>
void FMMMat<dim>::m2p_matvec(double* out, double* in) {
    for (size_t i = 0; i < m2p.obs_n_idx.size(); i++) {
        auto obs_n = obs_tree.nodes[m2p.obs_n_idx[i]];
        auto src_n = src_tree.nodes[m2p.src_n_idx[i]];

        auto equiv = inscribe_surf(src_n.bounds, cfg.inner_r, surf);
        interact_pts(
            cfg, out, in,
            &obs_tree.pts[obs_n.start], &obs_tree.normals[obs_n.start],
            obs_n.end - obs_n.start, obs_n.start,
            equiv.data(), surf.data(),
            surf.size(), src_n.idx * surf.size()
        );
    }
}

template <size_t dim>
void FMMMat<dim>::uc2e_matvec(double* out, double* in, int level) {
    auto& op = uc2e_ops[level];
    int n_rows = cfg.tensor_dim() * surf.size();
    for (size_t i = 0; i < uc2e[level].src_n_idx.size(); i++) {
        auto node_idx = m2p.src_n_idx[i];
        matrix_vector_product(
            op.data(), n_rows, n_rows, 
            &in[node_idx * n_rows],
            &out[node_idx * n_rows]
        );
    }
}

template <typename T>
void inplace_add_vecs(std::vector<T>& a, const std::vector<T>& b) {
    for (size_t j = 0; j < a.size(); j++) {
        a[j] += b[j];
    }
}

template <typename T>
void zero_vec(std::vector<T>& v) {
    std::fill(v.begin(), v.end(), 0.0);
}

template <size_t dim>
std::vector<double> FMMMat<dim>::p2p_eval(double* in) {
    auto n_outputs = obs_tree.pts.size() * tensor_dim();
    std::vector<double> out(n_outputs, 0.0);
    p2p_matvec(out.data(), in);
    return out;
}

template <size_t dim>
std::vector<double> FMMMat<dim>::p2m_eval(double* in) {
    auto n_multipoles = surf.size() * src_tree.nodes.size() * tensor_dim();
    std::vector<double> m_check(n_multipoles, 0.0);
    p2m_matvec(m_check.data(), in);

    std::vector<double> multipoles(n_multipoles, 0.0);
    // auto multipoles = uc2e_old[0].matvec(m_check.data(), n_multipoles);
    std::cout << "nmulti" << n_multipoles << std::endl;
    uc2e_matvec(multipoles.data(), m_check.data(), 0);

    for (size_t i = 1; i < m2m.size(); i++) {
        zero_vec(m_check);
        m2m_matvec(m_check.data(), multipoles.data(), i);
        uc2e_matvec(multipoles.data(), m_check.data(), i);
        // auto add_to_multipoles = uc2e_old[i].matvec(m_check.data(), n_multipoles);
        // inplace_add_vecs(multipoles, add_to_multipoles);
    }
    return multipoles;
}

template <size_t dim>
std::vector<double> FMMMat<dim>::m2p_eval(double* multipoles) {
    auto n_outputs = obs_tree.pts.size() * tensor_dim();
    std::vector<double> out(n_outputs, 0.0);
    m2p_matvec(out.data(), multipoles);
    return out;
}

template <size_t dim>
void build_uc2e(FMMMat<dim>& mat) {
    mat.uc2e_ops.resize(mat.src_tree.max_height + 1);
#pragma omp parallel for
    for (int i = mat.src_tree.max_height; i >= 0; i--) {
        double width = mat.src_tree.root().bounds.width / std::pow(2.0, static_cast<double>(i));
        std::array<double,dim> center{};
        Cube<dim> bounds(center, width);
        auto pinv = c2e_solve(mat.surf, bounds, mat.cfg.outer_r, mat.cfg.inner_r, mat.cfg);
        mat.uc2e_ops[i] = pinv;
    }
}

template <size_t dim>
FMMMat<dim> fmmmmmmm(const Octree<dim>& obs_tree, const Octree<dim>& src_tree,
                const FMMConfig<dim>& cfg) {

    auto translation_surf = surrounding_surface<dim>(cfg.order);

    FMMMat<dim> mat(obs_tree, src_tree, cfg, translation_surf);

    mat.m2m.resize(mat.src_tree.max_height + 1);
    mat.uc2e.resize(mat.src_tree.max_height + 1);
    mat.uc2e_old.resize(mat.src_tree.max_height + 1);

    Timer t;
    build_uc2e(mat);
    t.report("build_uc2e");
    up_collect(mat, mat.src_tree.root());
    t.report("up_collect");
    traverse(mat, mat.obs_tree.root(), mat.src_tree.root());
    t.report("traverse");
    std::cout << traverse_touches << " " << up_collect_touches << std::endl;

    return mat;
}

template 
FMMMat<2> fmmmmmmm(const Octree<2>& obs_tree, const Octree<2>& src_tree, const FMMConfig<2>& cfg);
template 
FMMMat<3> fmmmmmmm(const Octree<3>& obs_tree, const Octree<3>& src_tree, const FMMConfig<3>& cfg);
template struct FMMMat<2>;
template struct FMMMat<3>;
