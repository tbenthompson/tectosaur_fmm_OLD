import numpy as np
import tectosaur_fmm.fmm_wrapper as fmm
from tectosaur.util.timer import Timer

t = Timer()
np.random.seed(10)

kernel = 'elasticH'
tensor_dim = 3
n = 100000
mac = 3.0
order = 100
params = [1.0, 0.25]

pts = np.random.rand(n, 3)
ns = np.random.rand(n, 3)
ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
input = np.random.rand(n * tensor_dim)
t.report('setup problem')

tree = fmm.three.Octree(pts, ns, order)
t.report('build tree')

orig_idxs = np.array(tree.orig_idxs)
input_tree = input.reshape((-1,tensor_dim))[orig_idxs,:].reshape(-1)
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
orig_idxs = np.array(tree.orig_idxs)
to_orig[orig_idxs,:] = output
t.report('map to input space')

