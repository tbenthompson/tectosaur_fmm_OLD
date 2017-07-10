import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

from tectosaur.util.timer import Timer
import tectosaur_fmm.fmm_wrapper as fmm

quiet_tests = False
def test_print(*args, **kwargs):
    if not quiet_tests:
        print(*args, **kwargs)

def rand_pts(n, source):
    return np.random.rand(n, 3)

def ellipse_pts(n, source):
    a = 4.0
    b = 1.0
    c = 1.0
    uv = np.random.rand(n, 2)
    uv[:, 0] = (uv[:, 0] * np.pi) - np.pi / 2
    uv[:, 1] = (uv[:, 1] * 2 * np.pi) - np.pi
    x = a * np.cos(uv[:, 0]) * np.cos(uv[:, 1])
    y = b * np.cos(uv[:, 0]) * np.sin(uv[:, 1])
    z = c * np.sin(uv[:, 0])
    return np.array([x, y, z]).T

def run_full(n, make_pts, mac, order, kernel, params, ocl = False):
    t = Timer()
    obs_pts = make_pts(n, False)
    obs_ns = make_pts(n, False)
    obs_ns /= np.linalg.norm(obs_ns, axis = 1)[:,np.newaxis]
    src_pts = make_pts(n + 1, True)
    src_ns = make_pts(n + 1, True)
    src_ns /= np.linalg.norm(src_ns, axis = 1)[:,np.newaxis]
    t.report('gen random data')

    obs_kd = fmm.three.Octree(obs_pts, obs_ns, order)
    src_kd = fmm.three.Octree(src_pts, src_ns, order)
    t.report('build trees')
    fmm_mat = fmm.three.fmmmmmmm(
        obs_kd, src_kd, fmm.three.FMMConfig(1.1, mac, order, kernel, params)
    )
    t.report('setup fmm')

    tdim = fmm_mat.tensor_dim
    input_vals = np.ones(src_pts.shape[0] * tdim)
    n_outputs = obs_pts.shape[0] * tdim

    if ocl:
        est = fmm.eval_ocl(fmm_mat, input_vals)
    else:
        est = fmm.eval_cpu(fmm_mat, input_vals)
    t.report('eval fmm')
    # est2 = fmm.mf_direct_eval(kernel, obs_pts, obs_ns, src_pts, src_ns, params, input_vals)
    # t.report('eval direct')

    return (
        np.array(obs_kd.pts), np.array(obs_kd.normals),
        np.array(src_kd.pts), np.array(src_kd.normals), est
    )

def check(est, correct, accuracy):
    rmse = np.sqrt(np.mean((est - correct) ** 2))
    rms_c = np.sqrt(np.mean(correct ** 2))
    test_print("L2ERR: " + str(rmse / rms_c))
    test_print("MEANERR: " + str(np.mean(np.abs(est - correct)) / rms_c))
    test_print("MAXERR: " + str(np.max(np.abs(est - correct)) / rms_c))
    test_print("MEANRELERR: " + str(np.mean(np.abs((est - correct) / correct))))
    test_print("MAXRELERR: " + str(np.max(np.abs((est - correct) / correct))))
    lhs = est / rms_c
    rhs = correct / rms_c
    np.testing.assert_almost_equal(lhs, rhs, accuracy)

def check_invr(obs_pts, _0, src_pts, _1, est, accuracy = 3):
    correct_matrix = 1.0 / (scipy.spatial.distance.cdist(obs_pts, src_pts))
    correct_matrix[np.isnan(correct_matrix)] = 0
    correct_matrix[np.isinf(correct_matrix)] = 0

    correct = correct_matrix.dot(np.ones(src_pts.shape[0]))
    check(est, correct, accuracy)

def test_ones():
    obs_pts, _, src_pts, _, est = run_full(5000, rand_pts, 0.5, 1, "one",[])
    assert(np.all(np.abs(est - 5001) < 1e-3))

def test_invr():
    check_invr(*run_full(5000, rand_pts, 2.6, 100, "invr", []))

def test_irregular():
    check_invr(*run_full(10000, ellipse_pts, 2.6, 35, "invr", []))

def test_tensor():
    obs_pts, _, src_pts, _, est = run_full(5000, rand_pts, 2.6, 35, "tensor_invr", [])
    for d in range(3):
        check_invr(obs_pts, _, src_pts, _, est[d::3] / 3.0)

def test_double_layer():
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        20000, rand_pts, 3.0, 45, "laplace_double", []
    )
    correct_mat = fmm.direct_eval(
        "laplace_double", obs_pts, obs_ns, src_pts, src_ns, []
    ).reshape((obs_pts.shape[0], src_pts.shape[0]))
    correct = correct_mat.dot(np.ones(src_pts.shape[0]))
    check(est, correct, 3)
