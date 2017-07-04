import numpy as np
from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper
from tectosaur_fmm.fmm_wrapper import direct_eval

def test_it():
    n_obs = 1000
    n_src = 1000
    obs_pts = np.random.rand(n_obs, 3)
    obs_ns = np.random.rand(n_obs, 3)
    src_pts = np.random.rand(n_src, 3)
    src_ns = np.random.rand(n_src, 3)
    input = np.random.rand(n_src * 3)

    for K in ['U', 'T', 'A', 'H']:
        print('starting ' + K)
        out_p2p_mat = direct_eval('elastic' + K, obs_pts, obs_ns, src_pts, src_ns, [1.0, 0.25])
        out_p2p = out_p2p_mat.reshape((n_obs * 3, n_src * 3)).dot(input)
        out_tct = farfield_pts_wrapper(K, obs_pts, obs_ns, src_pts, src_ns, input, 1.0, 0.25)
        error = np.abs((out_tct - out_p2p) / out_tct)
        print(np.max(error))
        # np.testing.assert_almost_equal(error, 0, 4)
        # import ipdb; ipdb.set_trace()
