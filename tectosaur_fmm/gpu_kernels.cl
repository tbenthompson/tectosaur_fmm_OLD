<%!
#TODO: IS THERE A WAY TO SHARE THINGS BETWEEN OCL SCRIPTS?
def dn(dim):
    return ['x', 'y', 'z'][dim]

from tectosaur_fmm.cfg import gpu_float_type
from tectosaur.kernels import fmm_kernels
%>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define Real ${gpu_float_type}

// Atomic floating point addition for opencl
// from: https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
// I was worried this would cause a significant decrease in performance, but
// it doesn't seem to cause any problems. Probably there are so few conflicts
// that it's totally fine.
float atomic_fadd(volatile __global float *addr, float val)
{
    union{
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32 = *addr;
    do{
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg(
            (volatile __global unsigned int *)addr, 
            expected.u32, next.u32
        );
    } while( current.u32 != expected.u32 );
    return current.f32;
}

<%def name="load_bounds(K, node_name, type, R)">
    Real ${node_name}_surf_radius = ${type}_n_width[${node_name}_idx] * ${R} * 
        sqrt((double)${K.spatial_dim});
    % for d in range(K.spatial_dim):
    Real ${node_name}_center${dn(d)} = ${type}_n_center[${node_name}_idx * ${K.spatial_dim} + ${d}];
    % endfor
</%def>


<%def name="load_pts(K, type, index)">
    % for d in range(K.spatial_dim):
        Real ${type}${dn(d)} = ${type}_pts[${index} * ${K.spatial_dim} + ${d}];
        % if (type == 'obs' and K.needs_obsn) or (type == 'src' and K.needs_srcn):
            Real n${type}${dn(d)} = ${type}_ns[${index} * ${K.spatial_dim} + ${d}];
        % endif
    % endfor
</%def>

<%def name="load_surf_pts(K, type, node_name, index)">
    % for d in range(K.spatial_dim):
        Real n${type}${dn(d)} = surf[${index} * ${K.spatial_dim} + ${d}];
        Real ${type}${dn(d)} = ${node_name}_surf_radius * n${type}${dn(d)} + 
            ${node_name}_center${dn(d)};
    % endfor
</%def>

<%def name="init_sum(K)">
    % for d in range(K.tensor_dim):
    Real sum${dn(d)} = 0.0;
    % endfor
</%def>

<%def name="call_kernel(K, check_r_zero)">
    % for d in range(K.spatial_dim):
        Real D${dn(d)} = src${dn(d)} - obs${dn(d)};
    % endfor
    Real r2 = Dx * Dx;
    % for d in range(1, K.spatial_dim):
        r2 += D${dn(d)} * D${dn(d)};
    % endfor

    % if check_r_zero:
    if (r2 == 0) {
        continue;
    }
    % endif

    ${K.vector_code}
</%def>

<%def name="output_sum(K, out_idx)">
    % for d in range(K.tensor_dim):
    {
        __global Real* dest = &out[(${out_idx}) * ${K.tensor_dim} + ${d}];
        atomic_fadd(dest, sum${dn(d)});
    }
    % endfor
</%def>

<%def name="load_input(K, in_idx)">
    % for d in range(K.tensor_dim):
    Real in${dn(d)} = in[(${in_idx}) * ${K.tensor_dim} + ${d}];
    % endfor
</%def>

<%def name="get_block_idx()">
    const int global_idx = get_global_id(0); 
    const int worker_idx = get_local_id(0);
    const int block_idx = (global_idx - worker_idx) / ${n_workers_per_block};
</%def>

<%def name="p2p_kernel(K)">
__kernel
void p2p_kernel_${K.name}${K.spatial_dim}(
        __global Real* out, __global Real* in, int n_blocks,
        __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_start, __global int* src_n_end,
        __global Real* obs_pts, __global Real* obs_ns,
        __global Real* src_pts, __global Real* src_ns, __global Real* params)
{
    ${get_block_idx()}

    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];
    int src_start = src_n_start[block_idx];
    int src_end = src_n_end[block_idx];

    ${K.constants_code}

    for (int i = obs_start + worker_idx; i < obs_end; i += ${n_workers_per_block}) {
        ${load_pts(K, "obs", "i")}
        ${init_sum(K)}

        for (int j = src_start; j < src_end; j++) {
            ${load_pts(K, "src", "j")}
            ${load_input(K, "j")}
            ${call_kernel(K, True)}
        }

        ${output_sum(K, "i")}
    }
}
</%def>

<%def name="m2p_kernel(K)">
__kernel
void m2p_kernel_${K.name}${K.spatial_dim}(__global Real* out, __global Real* in,
        int n_blocks, __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_idx, int n_surf, __global Real* surf,
        __global Real* src_n_center, __global Real* src_n_width, Real inner_r,
        __global Real* obs_pts, __global Real* obs_ns, __global Real* params)
{
    ${get_block_idx()}

    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];
    int src_idx = src_n_idx[block_idx];

    ${load_bounds(K, "src", "src", "inner_r")}

    ${K.constants_code}

    for (int i = obs_start + worker_idx; i < obs_end; i += ${n_workers_per_block}) {
        ${load_pts(K, "obs", "i")}
        ${init_sum(K)}

        for (int j = 0; j < n_surf; j++) {
            ${load_surf_pts(K, "src", "src", "j")}
            ${load_input(K, "src_idx * n_surf + j")}
            ${call_kernel(K, False)}
        }

        ${output_sum(K, "i")}
    }
}
</%def>

<%def name="p2m_kernel(K)">
__kernel
void p2m_kernel_${K.name}${K.spatial_dim}(__global Real* out, __global Real* in,
        int n_blocks, __global int* parent_n_start, __global int* parent_n_end,
        __global int* parent_n_idx, int n_surf, __global Real* surf,
        __global Real* src_n_center, __global Real* src_n_width, Real outer_r,
        __global Real* src_pts, __global Real* src_ns, __global Real* params)
{
    ${get_block_idx()}

    int src_start = parent_n_start[block_idx];
    int src_end = parent_n_end[block_idx];
    int parent_idx = parent_n_idx[block_idx];

    ${load_bounds(K, "parent", "src", "outer_r")}

    ${K.constants_code}

    for (int i = worker_idx; i < n_surf; i += ${n_workers_per_block}) {
        ${load_surf_pts(K, "obs", "parent", "i")}
        ${init_sum(K)}

        for (int j = src_start; j < src_end; j++) {
            ${load_pts(K, "src", "j")}
            ${load_input(K, "j")}
            ${call_kernel(K, False)}
        }

        ${output_sum(K, "parent_idx * n_surf + i")}
    }
}
</%def>

<%def name="m2m_kernel(K)">
__kernel
void m2m_kernel_${K.name}${K.spatial_dim}(__global Real* out, __global Real* in,
        int n_blocks, __global int* parent_n_idx, __global int* child_n_idx,
        int n_surf, __global Real* surf,
        __global Real* src_n_center, __global Real* src_n_width,
        Real inner_r, Real outer_r, __global Real* params)
{
    ${get_block_idx()}

    int parent_idx = parent_n_idx[block_idx];
    int child_idx = child_n_idx[block_idx];

    ${load_bounds(K, "child", "src", "inner_r")}
    ${load_bounds(K, "parent", "src", "outer_r")}

    ${K.constants_code}

    for (int i = worker_idx; i < n_surf; i += ${n_workers_per_block}) {
        ${load_surf_pts(K, "obs", "parent", "i")}
        ${init_sum(K)}

        for (int j = 0; j < n_surf; j++) {
            ${load_surf_pts(K, "src", "child", "j")}
            ${load_input(K, "child_idx * n_surf + j")}
            ${call_kernel(K, False)}
        }

        ${output_sum(K, "parent_idx * n_surf + i")}
    }
}
</%def>

<%def name="uc2e_kernel(K)">
__kernel
void uc2e_kernel_${K.name}${K.spatial_dim}(__global Real* out, __global Real* in,
        int n_blocks, int n_rows, __global int* node_idx, __global int* node_depth,
        __global Real* ops)
{
    ${get_block_idx()}

    int n_idx = node_idx[block_idx];
    __global Real* op_start = &ops[node_depth[n_idx] * n_rows * n_rows];

    for (int i = worker_idx; i < n_rows; i += ${n_workers_per_block}) {
        Real sum = 0.0;
        for (int j = 0; j < n_rows; j++) {
            sum += op_start[i * n_rows + j] * in[n_idx * n_rows + j];
        }
        out[n_idx * n_rows + i] += sum;
    }
}
</%def>

% for K in fmm_kernels:
    ${p2p_kernel(K)}
    ${m2p_kernel(K)}
    ${p2m_kernel(K)}
    ${m2m_kernel(K)}
    ${uc2e_kernel(K)}
% endfor
