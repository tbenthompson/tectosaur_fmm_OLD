#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "include/timing.hpp"
#include "fmm_impl.hpp"
#include "blas_wrapper.hpp"

std::vector<std::array<double,3>> inscribe_surf(const Cube<3>& b, double scaling,
                                const std::vector<std::array<double,3>>& fmm_surf) {
    std::vector<std::array<double,3>> out(fmm_surf.size());
    for (size_t i = 0; i < fmm_surf.size(); i++) {
        for (size_t d = 0; d < 3; d++) {
            out[i][d] = fmm_surf[i][d] * b.R() * scaling + b.center[d];
        }
    }
    return out;
}

std::vector<std::array<double,3>> surrounding_surface_sphere(size_t order)
{
    std::vector<std::array<double,3>> pts;
    double a = 4 * M_PI / order;
    double d = std::sqrt(a);
    auto M_theta = static_cast<size_t>(std::round(M_PI / d));
    double d_theta = M_PI / M_theta;
    double d_phi = a / d_theta;
    for (size_t m = 0; m < M_theta; m++) {
        double theta = M_PI * (m + 0.5) / M_theta;
        auto M_phi = static_cast<size_t>(
            std::round(2 * M_PI * std::sin(theta) / d_phi)
        );
        for (size_t n = 0; n < M_phi; n++) {
            double phi = 2 * M_PI * n / M_phi;
            double x = std::sin(theta) * std::cos(phi);
            double y = std::sin(theta) * std::sin(phi);
            double z = std::cos(theta);
            pts.push_back({x, y, z});
        }
    }

    return pts;
}

extern "C" void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A,
                       int* LDA, double* X, int* INCX, double* BETA, double* Y,
                       int* INCY);
std::vector<double> BlockSparseMat::matvec(double* vec, size_t out_size) {
    char transpose = 'T';
    double alpha = 1;
    double beta = 1;
    int inc = 1;
    std::vector<double> out(out_size, 0.0);
    for (size_t b_idx = 0; b_idx < blocks.size(); b_idx++) {
        auto& b = blocks[b_idx];
        dgemv_(
            &transpose, &b.n_cols, &b.n_rows, &alpha, &vals[b.data_start],
            &b.n_cols, &vec[b.col_start], &inc, &beta, &out[b.row_start], &inc
        );
    }
    return out;
}

void MatrixFreeOp::insert(const OctreeNode<3>& obs_n, const OctreeNode<3>& src_n) {
    obs_n_start.push_back(obs_n.start);
    obs_n_end.push_back(obs_n.end);
    obs_n_idx.push_back(obs_n.idx);
    src_n_start.push_back(src_n.start);
    src_n_end.push_back(src_n.end);
    src_n_idx.push_back(src_n.idx);
}

void p2p(FMMMat& mat, const OctreeNode<3>& obs_n, const OctreeNode<3>& src_n) {
    mat.p2p.insert(obs_n, src_n);
}

void m2p(FMMMat& mat, const OctreeNode<3>& obs_n, const OctreeNode<3>& src_n) {
    mat.m2p.insert(obs_n, src_n);
}

void p2m(FMMMat& mat, const OctreeNode<3>& src_n) {
    mat.p2m.insert(src_n, src_n);
}

void m2m(FMMMat& mat, const OctreeNode<3>& parent_n, const OctreeNode<3>& child_n) {
    mat.m2m[parent_n.height].insert(parent_n, child_n);
}

void traverse(FMMMat& mat, const OctreeNode<3>& obs_n, const OctreeNode<3>& src_n) {
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
        for (int i = 0; i < 8; i++) {
            traverse(mat, obs_n, mat.src_tree.nodes[src_n.children[i]]);
        }
    } else {
        for (int i = 0; i < 8; i++) {
            traverse(mat, mat.obs_tree.nodes[obs_n.children[i]], src_n);
        }
    }
}

extern "C" void dgelsy_(int* M, int* N, int* NRHS, double* A, int* LDA,
                        double* B, int* LDB, int* JPVT, double* RCOND,
                        int* RANK, double* WORK, int* LWORK, int* INFO);

std::vector<double> qr_pseudoinverse(double* matrix, int n) {
    std::vector<double> rhs(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        rhs[i * n + i] = 1.0;
    }

    std::vector<int> jpvt(n, 0);
    int lwork = 4 * n + 1;
    std::vector<double> work(lwork);
    int rank;
    int info;
    double rcond = 1e-15;
    dgelsy_(&n, &n, &n, matrix, &n, rhs.data(), &n, jpvt.data(), &rcond, &rank,
            work.data(), &lwork, &info);
    return rhs;
}

// invert equiv to check operator
// In some cases, the equivalent surface to check surface operator
// is poorly conditioned. In this case, truncate the singular values
// to solve a regularized least squares version of the problem.
//
// TODO: There is quite a bit of numerical error incurred by storing this
// fully inverted and truncated.
//
// Can I just store it in factored form? Increases complexity.
// Without doing this, the error is can't be any lower than ~10^-10. Including
// this, the error can get down to 10^-15.
// I don't expect to need anything better than 10^-10. But in single precision,
// the number may be much lower. In which case, I may need to go to double
// precision sooner than I'd prefer.
// So, I see two ways to design this. I can store the check to equiv matrix
// along with each block that needs it. Or, I can separate the P2M, M2M, M2L,
// P2L, L2L into two steps each: P2M, M2M, M2L, P2L, L2L and UC2E and DC2E
// (Up check to equiv and down check to equiv)
// The latter approach seems better, since less needs to be stored. The
// U2M and D2L matrices should be separated by level like M2M and L2L.
// <-- (later note) I did this.
void c2e(FMMMat& mat, BlockSparseMat& sub_mat, const OctreeNode<3>& node,
         double check_r, double equiv_r) {
    auto equiv_surf = mat.get_surf(node, equiv_r);
    auto check_surf = mat.get_surf(node, check_r);
    auto n_surf = mat.surf.size();

    auto n_rows = n_surf * mat.tensor_dim();

    std::vector<double> equiv_to_check(n_rows * n_rows);
    mat.cfg.kernel.f(
        {
            check_surf.data(), mat.surf.data(), 
            equiv_surf.data(), mat.surf.data(),
            n_surf, n_surf,
            mat.cfg.params.data()
        },
        equiv_to_check.data());

    // TODO: Currently, svd decomposition is the most time consuming part of
    // assembly. How to optimize this?
    // 1) Batch a bunch of SVDs to the gpu.
    // 2) Figure out a way to do less of them. Prune tree nodes?
    //   May get about 25-50% faster.
    // 3) A faster alternative? QR? <--- This seems like the first step.
    // 4) BEST OPTIONBEST OPTIONBEST OPTIONBEST OPTIONBEST OPTION: Regular octree
    // so that the number of stored c2e operators can be small.
    // auto svd = svd_decompose(equiv_to_check.data(), n_rows);
    // const double truncation_threshold = 1e-15;
    // set_threshold(svd, truncation_threshold);
    // auto pinv = svd_pseudoinverse(svd);
    auto pinv = qr_pseudoinverse(equiv_to_check.data(), n_rows);

    sub_mat.blocks.push_back({node.idx * n_rows, node.idx * n_rows, int(n_rows),
                          int(n_rows), sub_mat.vals.size()});
    sub_mat.vals.insert(sub_mat.vals.end(), pinv.begin(), pinv.end());
}

void up_collect(FMMMat& mat, const OctreeNode<3>& src_n) {
    c2e(mat, mat.uc2e[src_n.height], src_n, mat.cfg.outer_r, mat.cfg.inner_r);
    if (src_n.is_leaf) {
        p2m(mat, src_n);
    } else {
        for (int i = 0; i < 8; i++) {
            auto child = mat.src_tree.nodes[src_n.children[i]];
            up_collect(mat, child);
            m2m(mat, src_n, child);
        }
    }
}

FMMMat::FMMMat(Octree<3> obs_tree, Octree<3> src_tree, FMMConfig cfg,
        std::vector<std::array<double,3>> surf):
    obs_tree(obs_tree),
    src_tree(src_tree),
    cfg(cfg),
    surf(surf),
    translation_surface_order(surf.size())
{}

void interact_pts(const FMMConfig& cfg, double* out, double* in,
    const std::array<double,3>* obs_pts, const std::array<double,3>* obs_ns,
    size_t n_obs, size_t obs_pt_start,
    const std::array<double,3>* src_pts, const std::array<double,3>* src_ns,
    size_t n_src, size_t src_pt_start) 
{
    if (n_obs == 0 || n_src == 0) {
        return;
    }

    double* out_val_start = &out[cfg.tensor_dim() * obs_pt_start];
    double* in_val_start = &in[cfg.tensor_dim() * src_pt_start];
    cfg.kernel.f_mf(
        NBodyProblem{obs_pts, obs_ns, src_pts, src_ns, n_obs, n_src, cfg.params.data()},
        out_val_start, in_val_start
    );
}

std::vector<std::array<double,3>> FMMMat::get_surf(const OctreeNode<3>& src_n, double r) {
    return inscribe_surf(src_n.bounds, r, surf);
}

void FMMMat::p2p_matvec(double* out, double* in) {
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

void FMMMat::p2m_matvec(double* out, double *in) {
    for (size_t i = 0; i < p2m.obs_n_idx.size(); i++) {
        auto src_n = src_tree.nodes[p2m.src_n_idx[i]];
        auto check = get_surf(src_n, cfg.outer_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), src_n.idx * surf.size(),
            &src_tree.pts[src_n.start], &src_tree.normals[src_n.start],
            src_n.end - src_n.start, src_n.start
        );
    }
}

void FMMMat::m2m_matvec(double* out, double *in, int level) {
    for (size_t i = 0; i < m2m[level].obs_n_idx.size(); i++) {
        auto parent_n = src_tree.nodes[m2m[level].obs_n_idx[i]];
        auto child_n = src_tree.nodes[m2m[level].src_n_idx[i]];
        auto check = get_surf(parent_n, cfg.outer_r);
        auto equiv = get_surf(child_n, cfg.inner_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), parent_n.idx * surf.size(),
            equiv.data(), surf.data(), 
            surf.size(), child_n.idx * surf.size()
        );
    }
}

void FMMMat::m2p_matvec(double* out, double* in) {
    for (size_t i = 0; i < m2p.obs_n_idx.size(); i++) {
        auto obs_n = obs_tree.nodes[m2p.obs_n_idx[i]];
        auto src_n = src_tree.nodes[m2p.src_n_idx[i]];

        auto equiv = get_surf(src_n, cfg.inner_r);
        interact_pts(
            cfg, out, in,
            &obs_tree.pts[obs_n.start], &obs_tree.normals[obs_n.start],
            obs_n.end - obs_n.start, obs_n.start,
            equiv.data(), surf.data(),
            surf.size(), src_n.idx * surf.size()
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

std::vector<double> FMMMat::p2p_eval(double* in) {
    auto n_outputs = obs_tree.pts.size() * tensor_dim();
    std::vector<double> out(n_outputs, 0.0);
    p2p_matvec(out.data(), in);
    return out;
}

std::vector<double> FMMMat::p2m_eval(double* in) {
    auto n_multipoles = surf.size() * src_tree.nodes.size() * tensor_dim();
    std::vector<double> m_check(n_multipoles, 0.0);
    p2m_matvec(m_check.data(), in);

    auto multipoles = uc2e[0].matvec(m_check.data(), n_multipoles);

    for (size_t i = 1; i < m2m.size(); i++) {
        zero_vec(m_check);
        m2m_matvec(m_check.data(), multipoles.data(), i);
        auto add_to_multipoles = uc2e[i].matvec(m_check.data(), n_multipoles);
        inplace_add_vecs(multipoles, add_to_multipoles);
    }
    return multipoles;
}

std::vector<double> FMMMat::m2p_eval(double* multipoles) {
    auto n_outputs = obs_tree.pts.size() * tensor_dim();
    std::vector<double> out(n_outputs, 0.0);
    m2p_matvec(out.data(), multipoles);
    return out;
}

FMMMat fmmmmmmm(const Octree<3>& obs_tree, const Octree<3>& src_tree,
                const FMMConfig& cfg) {

    auto translation_surf = surrounding_surface_sphere(cfg.order);

    FMMMat mat(obs_tree, src_tree, cfg, translation_surf);

    mat.m2m.resize(mat.src_tree.max_height + 1);
    mat.uc2e.resize(mat.src_tree.max_height + 1);

#pragma omp parallel
#pragma omp single nowait
    {
#pragma omp task
        up_collect(mat, mat.src_tree.root());
#pragma omp task
        traverse(mat, mat.obs_tree.root(), mat.src_tree.root());
    }

    return mat;
}
