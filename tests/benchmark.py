import numpy as np
import tectosaur_fmm.fmm_wrapper as fmm
from tectosaur.util.timer import Timer
from tectosaur.util.test_decorators import golden_master

def runner():
    t = Timer()
    np.random.seed(10)

    kernel = 'elasticH'
    tensor_dim = 3
    n = 1000000
    mac = 3.0
    order = 150
    pts_per_cell = order
    # pts_per_cell = n + 1
    params = [1.0, 0.25]

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
        tree, tree, fmm.three.FMMConfig(1.1, mac, order, kernel, params)
    )
    t.report('setup fmm')
    fmm.report_p2p_vs_m2p(fmm_mat)
    t.report('report')

    output = fmm.eval_ocl(fmm_mat, input_tree)
    t.report('eval fmm')

    output = output.reshape((-1, tensor_dim))
    to_orig = np.empty_like(output)
    to_orig[np.array(tree.orig_idxs),:] = output
    t.report('map to input space')
    return to_orig

@golden_master(6)
def test_benchmark(request):
    out = runner()
    return out / np.max(np.abs(out))

if __name__ == '__main__':
    runner()
