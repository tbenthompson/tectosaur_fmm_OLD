#include "octree.hpp"

template <size_t dim>
std::array<int,OctreeNode<dim>::split+1> octree_partition(
        const Cube<dim>& bounds, PtNormal<dim>* start, PtNormal<dim>* end) 
{
    std::array<std::vector<PtNormal<dim>>,OctreeNode<dim>::split> chunks{};
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
std::vector<PtNormal<dim>> combine_pts_normals(std::array<double,dim>* pts,
        std::array<double,dim>* normals, size_t n_pts) 
{
    std::vector<PtNormal<dim>> pts_normals(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        pts_normals[i] = {pts[i], normals[i], i};
    }
    return pts_normals;
}

template <size_t dim>
Octree<dim>::Octree(std::array<double,dim>* in_pts, std::array<double,dim>* in_normals,
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

template <size_t dim>
size_t Octree<dim>::add_node(size_t start, size_t end, 
    size_t n_per_cell, int depth, Cube<dim> bounds,
    std::vector<PtNormal<dim>>& temp_pts)
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

template std::vector<PtNormal<2>> combine_pts_normals(std::array<double,2>* pts,
        std::array<double,2>* normals, size_t n_pts);
template std::vector<PtNormal<3>> combine_pts_normals(std::array<double,3>* pts,
        std::array<double,3>* normals, size_t n_pts);

template struct Octree<2>;
template struct Octree<3>;
