import numpy as np
import tectosaur_fmm.fmm_wrapper as fmm

def test_kdtree_bisects():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    pts = np.array(kdtree.pts)
    for n in kdtree.nodes:
        if n.is_leaf:
            continue
        l = kdtree.nodes[n.children[0]]
        r = kdtree.nodes[n.children[1]]
        assert(l.start == n.start)
        assert(r.end == n.end)
        assert(l.end == r.start)

def test_kdtree_contains_pts():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    pts = np.array(kdtree.pts)
    for n in kdtree.nodes:
        for d in range(3):
            dist = np.sqrt(
                np.sum((pts[n.start:n.end,:] - n.bounds.center) ** 2, axis = 1)
            )
            assert(np.all(dist < n.bounds.r * 1.0001))

def test_kdtree_height_depth():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    for n in kdtree.nodes:
        if n.is_leaf:
            continue
        for c in range(2):
            assert(n.depth == kdtree.nodes[n.children[c]].depth - 1)
        assert(n.height ==
            max([kdtree.nodes[n.children[c]].height for c in range(2)]) + 1)

def test_kdtree_orig_idx():
    pts = np.random.rand(1000,3)
    kdtree = fmm.KDTree(pts, pts, 50)
    np.testing.assert_almost_equal(np.array(kdtree.pts), pts[np.array(kdtree.orig_idxs), :])

def test_kdtree_idx():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    for i, n in enumerate(kdtree.nodes):
        assert(n.idx == i)

