from tectosaur.util.test_decorators import slow

@slow
def test_self_fmm():
    order = 60
    mac = 3.0
    n = 15000
    k_name = "elasticU"
    params = [1.0, 0.25]
    pts = np.random.rand(n, 3)
    ns = np.random.rand(n, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    kd = fmm.KDTree(pts, ns, order)
    fmm_mat = fmm.fmmmmmmm(
        kd, kd, fmm.FMMConfig(1.1, mac, order, k_name, params)
    )
    est = fmm.eval(fmm_mat, np.ones(n * 3))
    correct_mat = fmm.direct_eval(
        k_name, np.array(kd.pts), np.array(kd.normals),
        np.array(kd.pts), np.array(kd.normals), params
    ).reshape((n * 3, n * 3))
    correct_mat[np.isnan(correct_mat)] = 0
    correct_mat[np.isinf(correct_mat)] = 0
    correct = correct_mat.dot(np.ones(n * 3))
    check(est, correct, 2)


@slow
def test_build_big():
    pts = np.random.rand(1000000, 3)
    import time
    start = time.time()
    kdtree = fmm.KDTree(pts, pts, 1)
    test_print("KDTree took: " + str(time.time() - start))

@slow
def test_high_accuracy():
    check_invr(*run_full(15000, rand_pts, 2.6, 200, "invr", []), accuracy = 8)

@slow
def test_elasticH():
    params = [1.0, 0.25]
    K = "elasticT"
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        10000, ellipse_pts, 2.8, 52, K, params
    )
    # correct_mat = fmm.direct_eval(
    #     K, obs_pts, obs_ns, src_pts, src_ns, params
    # ).reshape((3 * obs_pts.shape[0], 3 * src_pts.shape[0]))
    # correct = correct_mat.dot(np.ones(3 * src_pts.shape[0]))
    # check(est, correct, 3)

