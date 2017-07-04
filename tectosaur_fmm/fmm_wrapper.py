import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer

import tectosaur_fmm
from tectosaur_fmm.cfg import float_type

import cppimport
fmm = cppimport.imp("tectosaur_fmm.fmm").fmm
for k in dir(fmm):
    locals()[k] = getattr(fmm, k)

gpu_module = None

def get_gpu_module():
    global gpu_module
    if gpu_module is None:
        gpu_module = gpu.load_gpu(
            'gpu_kernels.cl',
            tmpl_dir = tectosaur_fmm.source_dir,
            no_caching = True
        )
    return gpu_module

def eval(fmm_mat, input_vals):
    t = Timer()

    k_name = fmm_mat.cfg.kernel_name()
    tensor_dim = fmm_mat.cfg.tensor_dim()
    module = get_gpu_module()
    p2p = getattr(module, 'p2p_kernel_' + k_name)
    m2p = getattr(module, 'm2p_kernel_' + k_name)

    t.report('get gpu fncs')

    #TODO: Benchmark and check if its worth exposing the
    # buffer interface for these arrays to avoid copying the data
    gpu_obs_pts = gpu.to_gpu(np.array(fmm_mat.obs_tree.pts), float_type)
    gpu_obs_normals = gpu.to_gpu(np.array(fmm_mat.obs_tree.normals), float_type)
    gpu_src_pts = gpu.to_gpu(np.array(fmm_mat.src_tree.pts), float_type)
    gpu_src_normals = gpu.to_gpu(np.array(fmm_mat.src_tree.normals), float_type)

    gpu_obs_n_start = gpu.to_gpu(np.array(fmm_mat.p2p.obs_n_start), np.int32)
    gpu_obs_n_end = gpu.to_gpu(np.array(fmm_mat.p2p.obs_n_end), np.int32)
    gpu_src_n_start = gpu.to_gpu(np.array(fmm_mat.p2p.src_n_start), np.int32)
    gpu_src_n_end = gpu.to_gpu(np.array(fmm_mat.p2p.src_n_end), np.int32)

    gpu_out = gpu.zeros_gpu(tensor_dim * gpu_obs_pts.shape[0], float_type)
    gpu_in = gpu.to_gpu(input_vals, float_type)
    t.report("data to gpu")

    multipoles = fmm_mat.p2m_eval(input_vals)
    t.report('p2m')

    p2p(
        gpu.gpu_queue, (gpu_obs_n_start.shape[0],), None,
        gpu_out.data, gpu_in.data,
        gpu_obs_n_start.data, gpu_obs_n_end.data,
        gpu_src_n_start.data, gpu_src_n_end.data,
        gpu_obs_pts.data, gpu_obs_normals.data,
        gpu_src_pts.data, gpu_src_normals.data
    )
    t.report('launch p2p')


    gpu_obs_n_start = gpu.to_gpu(np.array(fmm_mat.m2p.obs_n_start), np.int32)
    gpu_obs_n_end = gpu.to_gpu(np.array(fmm_mat.m2p.obs_n_end), np.int32)

    gpu_src_n_idx = gpu.to_gpu(np.array(fmm_mat.m2p.src_n_idx), np.int32)
    gpu_centers = gpu.to_gpu(
        np.array([n.bounds.center for n in fmm_mat.src_tree.nodes]).flatten(),
        float_type
    )
    gpu_surf = gpu.to_gpu(np.array(fmm_mat.surf), float_type)
    gpu_rs = gpu.to_gpu(np.array([n.bounds.r for n in fmm_mat.src_tree.nodes]), float_type)

    gpu_multipoles = gpu.to_gpu(multipoles, float_type)
    t.report("m2p data to gpu")

    ends = np.array(fmm_mat.m2p.obs_n_end)
    starts = np.array(fmm_mat.m2p.obs_n_start)
    m2p_i = np.sum(len(fmm_mat.surf) * (ends - starts))
    obs_ends = np.array(fmm_mat.p2p.obs_n_end)
    obs_starts = np.array(fmm_mat.p2p.obs_n_start)
    src_ends = np.array(fmm_mat.p2p.src_n_end)
    src_starts = np.array(fmm_mat.p2p.src_n_start)
    p2p_i = np.sum((obs_ends - obs_starts) * (src_ends - src_starts))
    tree_i = p2p_i + m2p_i
    direct_i = gpu_obs_pts.shape[0] ** 2
    print(tree_i / direct_i, tree_i, p2p_i, m2p_i)

    n_m2p = gpu_obs_n_start.shape[0]
    if n_m2p > 0:
        m2p(
            gpu.gpu_queue, (n_m2p,), None,
            gpu_out.data, gpu_multipoles.data,
            gpu_obs_n_start.data, gpu_obs_n_end.data,
            gpu_src_n_idx.data, np.int32(len(fmm_mat.surf)), gpu_surf.data,
            gpu_centers.data, gpu_rs.data, float_type(fmm_mat.cfg.inner_r),
            gpu_obs_pts.data, gpu_obs_normals.data,
        )

    retval = gpu_out.get()
    t.report("m2p")
    return retval


