import numpy as np
from tectosaur.util.test_decorators import slow, kernel

import tectosaur_fmm.fmm_wrapper as fmm
from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper
from test_fmm import check_invr, run_full, rand_pts, check

@slow
def test_self_fmm():
# def test_self_fmm(kernel):
    kernel = 'elasticH'
    np.random.seed(10)
    n = 5000
    params = [1.0, 0.25]
    pts = np.random.rand(n, 3)
    ns = np.random.rand(n, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]

    mac = 3.0
    order = 40

    kd = fmm.KDTree(pts, ns, order)
    fmm_mat = fmm.fmmmmmmm(
        kd, kd, fmm.FMMConfig(1.1, mac, order, kernel, params)
    )
    tensor_dim = fmm_mat.cfg.tensor_dim()
    input = np.random.rand(n * tensor_dim)
    output = fmm.eval_ocl(fmm_mat, input)

    # output = output.reshape((-1, 3))
    # to_orig = np.empty_like(output)
    # orig_idxs = np.array(kd.orig_idxs, np.int64)
    # to_orig[orig_idxs,:] = output
    # results.append(to_orig)
    # results = np.array(results)


    correct_mat = fmm.direct_eval(
        kernel, np.array(kd.pts), np.array(kd.normals),
        np.array(kd.pts), np.array(kd.normals), params
    ).reshape((n * tensor_dim, n * tensor_dim))
    correct_mat[np.isnan(correct_mat)] = 0
    correct_mat[np.isinf(correct_mat)] = 0
    correct2 = correct_mat.dot(input)

    correct = farfield_pts_wrapper(
        kernel, np.array(kd.pts), np.array(kd.normals),
        np.array(kd.pts), np.array(kd.normals), input, params
    )
    check(output, correct, 2)
    check(correct2, correct, 2)

@slow
def test_build_big():
    pts = np.random.rand(1000000, 3)
    import time
    start = time.time()
    kdtree = fmm.KDTree(pts, pts, 1)
    print("KDTree took: " + str(time.time() - start))

@slow
def test_high_accuracy():
    import time
    start = time.time()
    # check_invr(*run_full(15000, rand_pts, 2.6, 100, "invr", []), accuracy = 6)
    run_full(300000, rand_pts, 2.6, 100, "invr", [])
    print("took: " + str(time.time() - start))

@slow
def test_elasticH():
    params = [1.0, 0.25]
    K = "elasticH"
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        10000, ellipse_pts, 2.8, 52, K, params
    )
    # correct_mat = fmm.direct_eval(
    #     K, obs_pts, obs_ns, src_pts, src_ns, params
    # ).reshape((3 * obs_pts.shape[0], 3 * src_pts.shape[0]))
    # correct = correct_mat.dot(np.ones(3 * src_pts.shape[0]))
    # check(est, correct, 3)

