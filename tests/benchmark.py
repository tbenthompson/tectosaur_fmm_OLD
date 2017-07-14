import numpy as np
import tectosaur_fmm.fmm_wrapper as fmm
from tectosaur.util.timer import Timer
from tectosaur.util.test_decorators import golden_master

from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper


K = 'elasticH'
tensor_dim = 3
n = 1000000
params = [1.0, 0.25]

fmm.get_gpu_module()

def direct_runner():
    t = Timer()
    np.random.seed(10)

    pts = np.random.rand(n, 3)
    ns = np.random.rand(n, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(n * tensor_dim)
    t.report('setup problem')

    out_direct = farfield_pts_wrapper(K, pts, ns, pts, ns, input, params)
    t.report('eval direct')

    return out_direct

def fmm_runner():
    t = Timer()
    np.random.seed(10)

    mac = 3.0
    order = 150
    pts_per_cell = order * 2
    # pts_per_cell = n + 1

    pts = np.random.rand(n, 3)
    ns = np.random.rand(n, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(n * tensor_dim)
    t.report('setup problem')

    tree = fmm.three.Octree(pts, ns, pts_per_cell)
    t.report('build tree')

    input_tree = input.reshape((-1,tensor_dim))[np.array(tree.orig_idxs),:].reshape(-1)
    t.report('map input to tree space')

    fmm_mat = fmm.three.fmmmmmmm(
        tree, tree, fmm.three.FMMConfig(1.1, mac, order, K, params)
    )
    t.report('setup fmm')
    fmm.report_p2p_vs_m2p(fmm_mat)
    t.report('report')

    gpu_data = fmm.data_to_gpu(fmm_mat, input_tree)
    t.report('data to gpu')

    output = fmm.eval_ocl(fmm_mat, input_tree, gpu_data)
    t.report('eval fmm')

    output = output.reshape((-1, tensor_dim))
    to_orig = np.empty_like(output)
    to_orig[np.array(tree.orig_idxs),:] = output
    t.report('map to input space')
    return to_orig

@golden_master(6)
def test_benchmark(request):
    global n
    n = 100000
    out = fmm_runner()
    print(out / np.max(np.abs(out)))
    return out / np.max(np.abs(out))

def compare_to_direct(A):
    B = direct_runner()
    L2B = np.sqrt(np.sum(B ** 2))
    L2Diff = np.sqrt(np.sum((A - B) ** 2))
    relL2 = L2Diff / L2B
    print(L2B, L2Diff, relL2)

if __name__ == '__main__':
    A = fmm_runner().flatten()
    # compare_to_direct(A)
