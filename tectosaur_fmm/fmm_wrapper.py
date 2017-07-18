import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer

import tectosaur_fmm
from tectosaur_fmm.cfg import float_type
from tectosaur.kernels import kernels

import cppimport
fmm = cppimport.imp("tectosaur_fmm.fmm").fmm

for k in dir(fmm):
    locals()[k] = getattr(fmm, k)

gpu_module = None

if 'profile' not in __builtins__:
    def profile(f):
        return f

n_workers_per_block = 128

def get_gpu_module(surf, K):
    args = dict(
        n_workers_per_block = n_workers_per_block,
        surf = surf,
        K = K
    )
    gpu_module = gpu.load_gpu(
        'gpu_kernels.cl',
        tmpl_dir = tectosaur_fmm.source_dir,
        tmpl_args = args
    )
    return gpu_module

def report_p2p_vs_m2p(fmm_mat):
    surf_n = len(fmm_mat.surf)

    starts = fmm_mat.p2m.obs_n_start
    ends = fmm_mat.p2m.obs_n_end
    p2m_i = np.sum((ends - starts) * surf_n)

    m2m_i = (
        sum([len(fmm_mat.m2m[level].obs_n_idx) for level in range(len(fmm_mat.m2m))])
        * surf_n * surf_n
    )

    ends = np.array(fmm_mat.m2p.obs_n_end)
    starts = np.array(fmm_mat.m2p.obs_n_start)
    m2p_i = np.sum((ends - starts) * surf_n)
    obs_ends = np.array(fmm_mat.p2p.obs_n_end)
    obs_starts = np.array(fmm_mat.p2p.obs_n_start)
    src_ends = np.array(fmm_mat.p2p.src_n_end)
    src_starts = np.array(fmm_mat.p2p.src_n_start)
    p2p_i = np.sum((obs_ends - obs_starts) * (src_ends - src_starts))
    tree_i = p2p_i + m2p_i + p2m_i + m2m_i
    direct_i = len(fmm_mat.obs_tree.pts) * len(fmm_mat.src_tree.pts)

    print('compression factor: ' + str(tree_i / direct_i))
    print('total tree interactions: %e' % tree_i)
    print('total p2p interactions: %e' % p2p_i)
    print('total p2m interactions: %e' % p2m_i)
    print('total m2m interactions: %e' % m2m_i)
    print('total m2p interactions: %e' % m2p_i)

def data_to_gpu(fmm_mat, input_vals):
    src_tree_nodes = fmm_mat.src_tree.nodes

    gd = dict()

    surf = np.array(fmm_mat.surf)
    K = kernels[fmm_mat.cfg.kernel_name]
    gd['module'] = get_gpu_module(surf, K)
    gd['obs_pts'] = gpu.to_gpu(fmm_mat.obs_tree.pts, float_type)
    gd['obs_normals'] = gpu.to_gpu(fmm_mat.obs_tree.normals, float_type)
    gd['src_pts'] = gpu.to_gpu(fmm_mat.src_tree.pts, float_type)
    gd['src_normals'] = gpu.to_gpu(fmm_mat.src_tree.normals, float_type)
    gd['dim'] = gd['obs_pts'].shape[1]
    gd['tensor_dim'] = fmm_mat.cfg.tensor_dim
    gd['n_surf_pts'] = np.int32(surf.shape[0])
    gd['n_surf_dofs'] = gd['n_surf_pts'] * gd['tensor_dim']
    gd['params'] = gpu.to_gpu(np.array(fmm_mat.cfg.params), float_type)
    gd['out'] = gpu.zeros_gpu(gd['tensor_dim'] * gd['obs_pts'].shape[0], float_type)
    gd['in'] = gpu.to_gpu(input_vals, float_type)
    gd['n_multipoles'] = gd['n_surf_dofs'] * fmm_mat.src_tree.n_nodes
    gd['m_check'] = gpu.zeros_gpu(gd['n_multipoles'], float_type)
    gd['multipoles'] = gpu.zeros_gpu(gd['n_multipoles'], float_type)

    gd['src_n_center'] = gpu.to_gpu(
        np.array([n.bounds.center for n in src_tree_nodes]).flatten(),
        float_type
    )
    gd['src_n_width'] = gpu.to_gpu(np.array(
        [n.bounds.width for n in src_tree_nodes]
    ), float_type)

    gd['p2p_obs_n_start'] = gpu.to_gpu(fmm_mat.p2p.obs_n_start, np.int32)
    gd['p2p_obs_n_end'] = gpu.to_gpu(fmm_mat.p2p.obs_n_end, np.int32)
    gd['p2p_src_n_start'] = gpu.to_gpu(fmm_mat.p2p.src_n_start, np.int32)
    gd['p2p_src_n_end'] = gpu.to_gpu(fmm_mat.p2p.src_n_end, np.int32)

    gd['p2m_parent_n_start'] = gpu.to_gpu(fmm_mat.p2m.src_n_start, np.int32)
    gd['p2m_parent_n_end'] = gpu.to_gpu(fmm_mat.p2m.src_n_end, np.int32)
    gd['p2m_parent_n_idx'] = gpu.to_gpu(fmm_mat.p2m.src_n_idx, np.int32)

    n_levels = len(fmm_mat.m2m)
    gd['m2m_parent_n_idx'] = [
        gpu.to_gpu(fmm_mat.m2m[level].obs_n_idx, np.int32) for level in range(n_levels)
    ]
    gd['m2m_child_n_idx'] = [
        gpu.to_gpu(fmm_mat.m2m[level].src_n_idx, np.int32) for level in range(n_levels)
    ]

    gd['uc2e_node_n_idx'] = [
        gpu.to_gpu(fmm_mat.uc2e[level].src_n_idx, np.int32) for level in range(n_levels)
    ]
    gd['uc2e_node_depth'] = gpu.to_gpu(np.array([n.depth for n in fmm_mat.src_tree.nodes]), np.int32)
    gd['uc2e_ops'] = gpu.to_gpu(fmm_mat.uc2e_ops, float_type)

    gd['m2p_obs_n_start'] = gpu.to_gpu(fmm_mat.m2p.obs_n_start, np.int32)
    gd['m2p_obs_n_end'] = gpu.to_gpu(fmm_mat.m2p.obs_n_end, np.int32)
    gd['m2p_src_n_idx'] = gpu.to_gpu(fmm_mat.m2p.src_n_idx, np.int32)
    return gd


def gpu_p2p(fmm_mat, gd):
    p2p = getattr(gd['module'], 'p2p_kernel_' + fmm_mat.cfg.kernel_name)
    n_p2p = gd['p2p_obs_n_start'].shape[0]
    return [p2p(
        gpu.gpu_queue,
        (n_p2p * n_workers_per_block,),
        (n_workers_per_block,),
        gd['out'].data, gd['in'].data,
        np.int32(n_p2p), gd['p2p_obs_n_start'].data, gd['p2p_obs_n_end'].data,
        gd['p2p_src_n_start'].data, gd['p2p_src_n_end'].data,
        gd['obs_pts'].data, gd['obs_normals'].data,
        gd['src_pts'].data, gd['src_normals'].data,
        gd['params'].data
    )]

def gpu_m2p(fmm_mat, gd, uc2e_ev):
    m2p = getattr(gd['module'], 'm2p_kernel_' + fmm_mat.cfg.kernel_name)
    n_m2p = gd['m2p_obs_n_start'].shape[0]
    if n_m2p > 0:
        return [m2p(
            gpu.gpu_queue,
            (n_m2p * n_workers_per_block,),
            (n_workers_per_block,),
            gd['out'].data, gd['multipoles'].data,
            np.int32(n_m2p), gd['m2p_obs_n_start'].data, gd['m2p_obs_n_end'].data,
            gd['m2p_src_n_idx'].data,
            gd['src_n_center'].data, gd['src_n_width'].data, float_type(fmm_mat.cfg.inner_r),
            gd['obs_pts'].data, gd['obs_normals'].data,
            gd['params'].data,
            wait_for = uc2e_ev
        )]
    else:
        return None


def gpu_p2m(fmm_mat, gd):
    p2m = getattr(gd['module'], 'p2m_kernel_' + fmm_mat.cfg.kernel_name)
    n_p2m = gd['p2m_parent_n_start'].shape[0]
    if n_p2m > 0:
        return [p2m(
            gpu.gpu_queue,
            (n_p2m * n_workers_per_block,),
            (n_workers_per_block,),
            gd['m_check'].data, gd['in'].data,
            np.int32(n_p2m), gd['p2m_parent_n_start'].data, gd['p2m_parent_n_end'].data,
            gd['p2m_parent_n_idx'].data,
            gd['src_n_center'].data, gd['src_n_width'].data, float_type(fmm_mat.cfg.outer_r),
            gd['src_pts'].data, gd['src_normals'].data,
            gd['params'].data
        )]
    else:
        return None

def gpu_m2m(fmm_mat, gd, level, uc2e_ev):
    m2m = getattr(gd['module'], 'm2m_kernel_' + fmm_mat.cfg.kernel_name)
    n_m2m = gd['m2m_parent_n_idx'][level].shape[0]
    if n_m2m > 0:
        return [m2m(
            gpu.gpu_queue,
            (n_m2m * n_workers_per_block,),
            (n_workers_per_block,),
            gd['m_check'].data, gd['multipoles'].data,
            np.int32(n_m2m), gd['m2m_parent_n_idx'][level].data, gd['m2m_child_n_idx'][level].data,
            gd['src_n_center'].data, gd['src_n_width'].data,
            float_type(fmm_mat.cfg.inner_r), float_type(fmm_mat.cfg.outer_r),
            gd['params'].data,
            wait_for = uc2e_ev
        )]
    else:
        return None

def gpu_uc2e(fmm_mat, gd, level, m2m_ev):
    uc2e = gd['module'].uc2e_kernel
    n_uc2e = gd['uc2e_node_n_idx'][level].shape[0]
    n_uc2e_rows = gd['tensor_dim'] * gd['n_surf_pts']
    if n_uc2e > 0:
        return [uc2e(
            gpu.gpu_queue,
            (n_uc2e * n_workers_per_block,),
            (n_workers_per_block,),
            gd['multipoles'].data, gd['m_check'].data,
            np.int32(n_uc2e), np.int32(n_uc2e_rows),
            gd['uc2e_node_n_idx'][level].data,
            gd['uc2e_node_depth'].data,
            gd['uc2e_ops'].data,
            wait_for = m2m_ev
        )]
    else:
        return None

def print_timing(p2p_ev, m2m_evs, uc2e_evs, m2p_ev):
    def get_time(ev):
        if ev is not None:
            return (ev[0].profile.end - ev[0].profile.start) * 1e-9
        return 0

    p2p_t = get_time(p2p_ev)
    m2p_t = get_time(m2p_ev)
    m2m_t = sum([get_time(level) for level in m2m_evs])
    uc2e_t = sum([get_time(level) for level in m2m_evs])
    print('p2p took ' + str(p2p_t))
    print('m2p took ' + str(m2p_t))
    print('m2m took ' + str(m2m_t))
    print('uc2e took ' + str(uc2e_t))

def eval_ocl(fmm_mat, input, gpu_data = None):
    if gpu_data is None:
        gpu_data = data_to_gpu(fmm_mat, input)

    p2p_ev = gpu_p2p(fmm_mat, gpu_data)

    m2m_evs = []
    uc2e_evs = []
    m2m_evs.append(gpu_p2m(fmm_mat, gpu_data))
    uc2e_evs.append(gpu_uc2e(fmm_mat, gpu_data, 0, m2m_evs[0]))

    for i in range(1, len(fmm_mat.m2m)):
        m2m_evs.append(gpu_m2m(fmm_mat, gpu_data, i, uc2e_evs[i - 1]))
        uc2e_evs.append(gpu_uc2e(fmm_mat, gpu_data, i, m2m_evs[i]))

    m2p_ev = gpu_m2p(fmm_mat, gpu_data, uc2e_evs[-1])
    if m2p_ev is not None:
        m2p_ev[0].wait()

    p2p_ev[0].wait()
    retval = gpu_data['out'].get()

    print_timing(p2p_ev, m2m_evs, uc2e_evs, m2p_ev)

    return retval


def eval_cpu(fmm_mat, input_vals):
    tensor_dim = fmm_mat.cfg.tensor_dim
    n_out = fmm_mat.obs_tree.pts.shape[0] * tensor_dim
    n_multipoles = fmm_mat.src_tree.n_nodes * len(fmm_mat.surf) * tensor_dim
    n_locals = fmm_mat.obs_tree.n_nodes * len(fmm_mat.surf) * tensor_dim

    out = np.zeros(n_out)
    fmm_mat.p2p_eval(out, input_vals)

    m_check = np.zeros(n_multipoles)
    multipoles = np.zeros(n_multipoles)

    fmm_mat.p2m_eval(m_check, input_vals)
    fmm_mat.uc2e_eval(multipoles, m_check, 0)

    for i in range(1, len(fmm_mat.m2m)):
        m_check[:] = 0 #TODO: Is this necessary?
        fmm_mat.m2m_eval(m_check, multipoles, i)
        fmm_mat.uc2e_eval(multipoles, m_check, i)

    l_check = np.zeros(n_locals)
    locals = np.zeros(n_locals)
    for i in range(0, len(fmm_mat.l2l)):
        print("hi " + str(i))
        fmm_mat.l2l_eval(l_check, locals, i)
        print("hi2 " + str(i))
        fmm_mat.dc2e_eval(locals, l_check, i)

    print("hi3")
    fmm_mat.l2p_eval(out, locals)

    fmm_mat.m2p_eval(out, multipoles)
    return out
