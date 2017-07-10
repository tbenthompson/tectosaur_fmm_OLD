#pragma once

#include <array>
#include <vector>
#include <memory>

template <size_t dim>
inline double dot(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    double out = 0;
    for (size_t d = 0; d < dim; d++) {
        out += a[d] * b[d];
    }
    return out;
}

template <size_t dim>
inline std::array<double,dim> sub(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    std::array<double,dim> out;
    for (size_t d = 0; d < dim; d++) {
        out[d] = a[d] - b[d];
    }
    return out;
}

template <size_t dim>
inline double hypot(const std::array<double,dim>& v) {
    return std::sqrt(dot(v, v));
}

template <size_t dim>
inline double dist(const std::array<double,dim>& a, const std::array<double,dim>& b) {
    return hypot(sub(a,b));
}

template <size_t dim>
struct Cube {
    std::array<double,dim> center;
    double width;

    double R() const {
        return width * std::sqrt(3.0);
    }
};

template <size_t dim>
Cube<dim> bounding_box(std::array<double,dim>* pts, size_t n_pts) {
    std::array<double,dim> center_of_mass{};
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            center_of_mass[d] += pts[i][d];
        }
    }
    for (size_t d = 0; d < dim; d++) {
        center_of_mass[d] /= n_pts;
    }

    double max_width = 0.0;
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            max_width = std::max(max_width, fabs(pts[i][d] - center_of_mass[d]));
        }
    }

    return {center_of_mass, max_width};
}

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

template <size_t dim>
Cube<dim> get_subcell(const Cube<dim>& b, const std::array<size_t,dim>& idx)
{
    auto new_width = b.width / 2.0;
    auto new_center = b.center;
    for (size_t d = 0; d < dim; d++) {
        new_center[d] += ((static_cast<double>(idx[d]) * 2) - 1) * new_width;
    }
    return {new_center, new_width};
}

template <size_t dim>
int find_containing_subcell(const Cube<dim>& b, const std::array<double,dim>& pt) {
    int child_idx = 0;
    for (size_t d = 0; d < dim; d++) {
        if (pt[d] > b.center[d]) {
            child_idx++; 
        }
        if (d < dim - 1) {
            child_idx = child_idx << 1;
        }
    }
    return child_idx;
}

template <size_t dim>
bool in_box(const Cube<dim>& b, const std::array<double,dim>& pt) {
    for (size_t d = 0; d < dim; d++) {
        if (fabs(pt[d] - b.center[d]) >= (1.0 + 1e-14) * b.width) {
            return false;
        }
    }
    return true;
}

template <size_t dim>
struct OctreeNode {
    static const size_t split = 2<<(dim-1);

    size_t start;
    size_t end;
    Cube<dim> bounds;
    bool is_leaf;
    int height;
    int depth;
    size_t idx;
    std::array<size_t,split> children;
};

template <size_t dim>
struct PtNormalREMOVE {
    std::array<double,dim> pt;
    std::array<double,dim> normal;
    size_t orig_idx;
};

template <size_t dim>
std::array<int,OctreeNode<dim>::split+1> octree_partition(
        const Cube<dim>& bounds, PtNormalREMOVE<dim>* start, PtNormalREMOVE<dim>* end) 
{
    std::array<std::vector<PtNormalREMOVE<dim>>,OctreeNode<dim>::split> chunks{};
    for (auto* entry = start; entry < end; entry++) {
        chunks[find_containing_subcell(bounds, entry->pt)].push_back(*entry);
    }

    auto* next = start;
    std::array<int,OctreeNode<dim>::split+1> splits{};
    for (size_t subcell_idx = 0; subcell_idx < OctreeNode<dim>::split; subcell_idx++) {
        size_t subcell_n_pts = chunks[subcell_idx].size();
        for (size_t i = 0; i < subcell_n_pts; i++) {
            *next = chunks[subcell_idx][i];
            next++;
        }
        splits[subcell_idx + 1] = splits[subcell_idx] + subcell_n_pts;
    }

    return splits;
}

template <size_t dim>
std::vector<PtNormalREMOVE<dim>> combine_pts_normals(std::array<double,dim>* pts,
        std::array<double,dim>* normals, size_t n_pts) 
{
    std::vector<PtNormalREMOVE<dim>> pts_normals(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        pts_normals[i] = {pts[i], normals[i], i};
    }
    return pts_normals;
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
        auto pts_normals = combine_pts_normals(in_pts, in_normals, n_pts);
        size_t n_leaves = n_pts / n_per_cell;

        // For n leaves in a binary tree, there should be ~2*n total nodes. This
        // will be a comfortable overestimate for an octree. TODO: Is this reserve worth
        // doing?
        nodes.reserve(2 * n_leaves);

        auto bounds = bounding_box(in_pts, n_pts);
        add_node(0, n_pts, n_per_cell, 0, bounds, pts_normals);

        max_height = nodes[0].height;

        for (size_t i = 0; i < n_pts; i++) {
            pts[i] = pts_normals[i].pt;
            normals[i] = pts_normals[i].normal;
            orig_idxs[i] = pts_normals[i].orig_idx;
        }
    }

    size_t add_node(size_t start, size_t end, 
        size_t n_per_cell, int depth, Cube<dim> bounds,
        std::vector<PtNormalREMOVE<dim>>& temp_pts)
    {
        bool is_leaf = end - start <= n_per_cell; 
        auto n_idx = nodes.size();
        nodes.push_back({start, end, bounds, is_leaf, 0, depth, n_idx, {}});
        if (!is_leaf) {
            auto splits = octree_partition(bounds, temp_pts.data() + start, temp_pts.data() + end);
            int max_child_height = 0;
            for (size_t octant = 0; octant < OctreeNode<dim>::split; octant++) {
                auto child_bounds = get_subcell(bounds, make_child_idx<dim>(octant));
                auto child_node_idx = add_node(
                    start + splits[octant], start + splits[octant + 1],
                    n_per_cell, depth + 1, child_bounds, temp_pts
                );
                nodes[n_idx].children[octant] = child_node_idx;
                max_child_height = std::max(max_child_height, nodes[child_node_idx].height);
            }
            nodes[n_idx].height = max_child_height + 1;
        }
        return n_idx;
    }
};
