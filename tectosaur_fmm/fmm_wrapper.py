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

def report_p2p_vs_m2p(fmm_mat):
    ends = np.array(fmm_mat.m2p.obs_n_end)
    starts = np.array(fmm_mat.m2p.obs_n_start)
    m2p_i = np.sum(len(fmm_mat.surf) * (ends - starts))
    obs_ends = np.array(fmm_mat.p2p.obs_n_end)
    obs_starts = np.array(fmm_mat.p2p.obs_n_start)
    src_ends = np.array(fmm_mat.p2p.src_n_end)
    src_starts = np.array(fmm_mat.p2p.src_n_start)
    p2p_i = np.sum((obs_ends - obs_starts) * (src_ends - src_starts))
    tree_i = p2p_i + m2p_i
    direct_i = len(fmm_mat.obs_tree.pts) * len(fmm_mat.src_tree.pts)

    print('compression factor: ' + str(tree_i / direct_i))
    print('total tree interactions: ' + str(tree_i))
    print('total p2p interactions: ' + str(p2p_i))
    print('total m2p interactions: ' + str(m2p_i))

def data_to_gpu_p2p(fmm_mat, input_vals):
    gd = dict()

    #TODO: Benchmark and check if its worth exposing the
    # buffer interface for these arrays to avoid copying the data
    gd['obs_pts'] = gpu.to_gpu(np.array(fmm_mat.obs_tree.pts), float_type)
    gd['obs_normals'] = gpu.to_gpu(np.array(fmm_mat.obs_tree.normals), float_type)
    gd['src_pts'] = gpu.to_gpu(np.array(fmm_mat.src_tree.pts), float_type)
    gd['src_normals'] = gpu.to_gpu(np.array(fmm_mat.src_tree.normals), float_type)

    gd['p2p_obs_n_start'] = gpu.to_gpu(np.array(fmm_mat.p2p.obs_n_start), np.int32)
    gd['p2p_obs_n_end'] = gpu.to_gpu(np.array(fmm_mat.p2p.obs_n_end), np.int32)
    gd['p2p_src_n_start'] = gpu.to_gpu(np.array(fmm_mat.p2p.src_n_start), np.int32)
    gd['p2p_src_n_end'] = gpu.to_gpu(np.array(fmm_mat.p2p.src_n_end), np.int32)

    gd['params'] = gpu.to_gpu(np.array(fmm_mat.cfg.params), float_type)
    gd['out'] = gpu.zeros_gpu(fmm_mat.cfg.tensor_dim() * gd['obs_pts'].shape[0], float_type)
    gd['in'] = gpu.to_gpu(input_vals, float_type)
    return gd

def data_to_gpu_m2p(fmm_mat, multipoles, gd):
    gd['surf'] = gpu.to_gpu(np.array(fmm_mat.surf), float_type)

    gd['m2p_obs_n_start'] = gpu.to_gpu(np.array(fmm_mat.m2p.obs_n_start), np.int32)
    gd['m2p_obs_n_end'] = gpu.to_gpu(np.array(fmm_mat.m2p.obs_n_end), np.int32)
    gd['m2p_src_n_idx'] = gpu.to_gpu(np.array(fmm_mat.m2p.src_n_idx), np.int32)

    gd['src_n_center'] = gpu.to_gpu(
        np.array([n.bounds.center for n in fmm_mat.src_tree.nodes]).flatten(),
        float_type
    )
    gd['src_n_width'] = gpu.to_gpu(np.array(
        [n.bounds.width for n in fmm_mat.src_tree.nodes]
    ), float_type)
    gd['multipoles'] = gpu.to_gpu(multipoles, float_type)

def gpu_p2p(fmm_mat, gd):
    p2p = getattr(get_gpu_module(), 'p2p_kernel_' + fmm_mat.cfg.kernel_name())
    n_p2p = gd['p2p_obs_n_start'].shape[0]
    p2p(
        gpu.gpu_queue, (n_p2p,), None,
        gd['out'].data, gd['in'].data,
        np.int32(n_p2p), gd['p2p_obs_n_start'].data, gd['p2p_obs_n_end'].data,
        gd['p2p_src_n_start'].data, gd['p2p_src_n_end'].data,
        gd['obs_pts'].data, gd['obs_normals'].data,
        gd['src_pts'].data, gd['src_normals'].data,
        gd['params'].data
    )

def gpu_m2p(fmm_mat, gd):
    m2p = getattr(get_gpu_module(), 'm2p_kernel_' + fmm_mat.cfg.kernel_name())
    n_m2p = gd['m2p_obs_n_start'].shape[0]
    if n_m2p > 0:
        m2p(
            gpu.gpu_queue, (n_m2p,), None,
            gd['out'].data, gd['multipoles'].data,
            np.int32(n_m2p), gd['m2p_obs_n_start'].data, gd['m2p_obs_n_end'].data,
            gd['m2p_src_n_idx'].data, np.int32(gd['surf'].shape[0]), gd['surf'].data,
            gd['src_n_center'].data, gd['src_n_width'].data, float_type(fmm_mat.cfg.inner_r),
            gd['obs_pts'].data, gd['obs_normals'].data,
            gd['params'].data
        )

def eval_ocl(fmm_mat, input_vals):
    t = Timer()

    gpu_data = data_to_gpu_p2p(fmm_mat, input_vals)
    t.report('data to gpu p2p')
    gpu_p2p(fmm_mat, gpu_data)
    t.report('launch p2p')

    multipoles = fmm_mat.p2m_eval(input_vals)
    t.report('p2m')

    data_to_gpu_m2p(fmm_mat, multipoles, gpu_data)
    t.report('data to gpu m2p')
    gpu_m2p(fmm_mat, gpu_data)
    retval = gpu_data['out'].get()
    t.report("m2p")

    return retval


def eval_cpu(fmm_mat, input_vals):
    report_p2p_vs_m2p(fmm_mat)
    p2p = fmm_mat.p2p_eval(input_vals)
    multipoles = fmm_mat.p2m_eval(input_vals)
    m2p = fmm_mat.m2p_eval(multipoles)
    return p2p + m2p
