#pragma once

#include <array>
#include <vector>
#include <memory>

template <size_t dim>
inline double dot(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <size_t dim>
inline std::array<double,dim> sub(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template <size_t dim>
inline double hypot(const std::array<double,dim>& v) {
    return std::sqrt(dot(v, v));
}

template <size_t dim>
struct Ball {
    std::array<double,dim> center;
    double r;
};

template <size_t dim>
struct OctreeNode {
    static const size_t split = 2<<(dim-1);

    size_t start;
    size_t end;
    Ball<dim> bounds;
    bool is_leaf;
    int height;
    int depth;
    size_t idx;
    std::array<size_t,split> children;
};

template <size_t dim>
struct PtNormal {
    std::array<double,dim> pt;
    std::array<double,dim> normal;
    size_t orig_idx;
};

template <size_t dim>
Ball<dim> node_bounds(PtNormal<dim>* pts, size_t n_pts, double parent_size) {
    std::array<double,dim> center_of_mass{};
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            center_of_mass[d] += pts[i].pt[d];
        }
    }
    for (size_t d = 0; d < dim; d++) {
        center_of_mass[d] /= n_pts;
    }

    double max_r = 0.0;
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            max_r = std::max(max_r, hypot(sub(pts[i].pt, center_of_mass)));
        }
    }

    // Limit sides to being 1 / 50 times the side length of the parent cell
    const static double side_ratio = 1.0 / 50.0;
    return {center_of_mass, std::max(parent_size * side_ratio, max_r)};
}


template <size_t dim>
struct Octree {
    std::vector<std::array<double,dim>> pts;
    std::vector<std::array<double,dim>> normals;
    std::vector<size_t> orig_idxs;

    size_t n_pts;
    int max_height;
    std::vector<OctreeNode<dim>> nodes;

    const OctreeNode<dim>& root() const { return nodes.front(); }

    Octree(std::array<double,dim>* in_pts, std::array<double,dim>* in_normals,
            size_t n_pts, size_t n_per_cell):
        pts(n_pts),
        normals(n_pts),
        orig_idxs(n_pts),
        n_pts(n_pts)
    {
        std::vector<PtNormal<dim>> pts_normals(n_pts);
        for (size_t i = 0; i < n_pts; i++) {
            pts_normals[i] = {in_pts[i], in_normals[i], i};
        }
        size_t n_leaves = n_pts / n_per_cell;

        // For n leaves in a binary tree, there should be ~2*n total nodes. This
        // will be a comfortable overestimate for an octree. TODO: Is this reserve worth
        // doing?
        nodes.reserve(2 * n_leaves);
        add_node(0, n_pts, 0, n_per_cell, 1.0, 0, pts_normals);
        max_height = nodes[0].height;

        for (size_t i = 0; i < n_pts; i++) {
            pts[i] = pts_normals[i].pt;
            normals[i] = pts_normals[i].normal;
            orig_idxs[i] = pts_normals[i].orig_idx;
        }
    }

    size_t add_node(size_t start, size_t end, int split_dim,
        size_t n_per_cell, double parent_size, int depth, std::vector<PtNormal<dim>>& temp_pts)
    {
        // auto bounds = kd_bounds(temp_pts.data() + start, end - start, parent_size);

        // if (end - start <= n_per_cell) {
        //     nodes.push_back({start, end, bounds, true, 0, depth, nodes.size(), {0, 0}});
        //     return nodes.back().idx;
        // } else {
        //     auto split = std::partition(
        //         temp_pts.data() + start, temp_pts.data() + end,
        //         [&] (const PtNormal& v) {
        //             return v.pt[split_dim] < bounds.center[split_dim]; 
        //         }
        //     );
        //     auto n_idx = nodes.size();
        //     nodes.push_back({start, end, bounds, false, 0, depth, n_idx, {0, 0}});
        //     auto l = add_node(
        //         start, split - temp_pts.data(), (split_dim + 1) % 3,
        //         n_per_cell, bounds.r, depth + 1, temp_pts
        //     );
        //     auto r = add_node(
        //         split - temp_pts.data(), end, (split_dim + 1) % 3,
        //         n_per_cell, bounds.r, depth + 1, temp_pts
        //     );
        //     nodes[n_idx].children = {l, r};
        //     nodes[n_idx].height = std::max(nodes[l].height, nodes[r].height) + 1;
        //     return n_idx;
        // }
        return 0;
    }
};

// template <size_t dim> struct Ball;
// template <size_t dim> struct Box;
// 
// template <size_t dim>
// struct Octree {
//     static const size_t split = 2<<(dim-1);
//     typedef std::array<std::unique_ptr<Octree<dim>>,split> ChildrenType;
//     const Box<dim> bounds;
//     const Box<dim> true_bounds;
//     const std::vector<size_t> indices;
//     const size_t level;
//     const size_t index;
//     ChildrenType children;
// 
//     size_t n_immediate_children() const; 
//     size_t n_children() const; 
//     bool is_leaf() const; 
//     size_t find_closest_nonempty_child(const Vec<double,dim>& pt) const;
// };
// 

template <size_t dim>
std::array<size_t,dim> make_child_idx(size_t i) 
{
    std::array<size_t,dim> child_idx;
    for (int d = dim - 1; d >= 0; d--) {
        auto idx = i % 2;
        i = i >> 1;
        child_idx[d] = idx;
    }
    return child_idx;
}

// 
// template <size_t dim>
// Octree<dim>
// make_octree(const std::vector<Ball<dim>>& pts, size_t min_pts_per_cell);
// 
// template <size_t dim>
// Octree<dim> 
// make_octree(const std::vector<Vec<double,dim>>& pts, size_t min_pts_per_cell);
