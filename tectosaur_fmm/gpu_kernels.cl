<%!
#TODO: IS THERE A WAY TO SHARE THINGS BETWEEN OCL SCRIPTS?
def dn(dim):
    return ['x', 'y', 'z'][dim]

from tectosaur_fmm.cfg import gpu_float_type
if gpu_float_type == 'float':
    real_sized_int = 'uint'
    atomic_cmpxchg_f = 'atomic_cmpxchg'
else:
    real_sized_int = 'ulong'
    atomic_cmpxchg_f = 'atom_cmpxchg'
%>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define Real ${gpu_float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

// Atomic floating point addition for opencl
// from: https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
// I was worried this would cause a significant decrease in performance, but
// it doesn't seem to cause any problems. Probably there are so few conflicts
// that it's totally fine.
void atomic_fadd(volatile __global Real *addr, Real val) {
    union {
        ${real_sized_int} u;
        Real f;
    } next, expected, current;
    current.f = *addr;
    do {
        expected.f = current.f;
        next.f = expected.f + val;
        current.u = ${atomic_cmpxchg_f}(
            (volatile __global ${real_sized_int} *)addr, expected.u, next.u
        );
    } while(current.u != expected.u);
}

<%def name="p2p_kernel(k_name, tensor_dim)">
__kernel
void p2p_kernel_${k_name}(__global Real* out, __global Real* in,
        __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_start, __global int* src_n_end,
        __global Real* obs_pts, __global Real* obs_ns,
        __global Real* src_pts, __global Real* src_ns)
{
    const int block_idx = get_global_id(0);
    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];
    int src_start = src_n_start[block_idx];
    int src_end = src_n_end[block_idx];

    ${prim.constants(k_name)}

    for (int i = obs_start; i < obs_end; i++) {
        % for d in range(3):
            Real x${dn(d)} = obs_pts[i * 3 + ${d}];
            Real n${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor

        % for d in range(tensor_dim):
        Real sum${dn(d)} = 0.0;
        % endfor

        for (int j = src_start; j < src_end; j++) {
            % for d in range(3):
                Real y${dn(d)} = src_pts[j * 3 + ${d}];
                Real l${dn(d)} = src_ns[j * 3 + ${d}];
            % endfor

            % for d in range(tensor_dim):
            Real in${dn(d)} = in[j * ${tensor_dim} + ${d}];
            % endfor

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            ${prim.vector_kernels(k_name)}
        }

        % for d in range(tensor_dim):
        atomic_fadd(&out[i * ${tensor_dim} + ${d}], sum${dn(d)});
        % endfor
    }
}
</%def>

<%def name="m2p_kernel(k_name, tensor_dim)">
__kernel
void m2p_kernel_${k_name}(__global Real* out, __global Real* in,
        __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_idx, int multipoles_per_cell, __global Real* surf,
        __global Real* src_n_center, __global Real* src_n_radius, Real inner_r,
        __global Real* obs_pts, __global Real* obs_ns)
{
    const int block_idx = get_global_id(0);
    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];

    int src_idx = src_n_idx[block_idx];

    Real surf_radius = src_n_radius[src_idx] * inner_r;
    % for d in range(3):
    Real center${dn(d)} = src_n_center[src_idx * 3 + ${d}];
    % endfor

    ${prim.constants(k_name)}

    for (int i = obs_start; i < obs_end; i++) {
        % for d in range(3):
            Real x${dn(d)} = obs_pts[i * 3 + ${d}];
            Real n${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor

        % for d in range(tensor_dim):
        Real sum${dn(d)} = 0.0;
        % endfor

        for (int j = 0; j < multipoles_per_cell; j++) {
            % for d in range(3):
                Real l${dn(d)} = surf[j * 3 + ${d}];
                Real y${dn(d)} = surf_radius * l${dn(d)} + center${dn(d)};
            % endfor

            % for d in range(tensor_dim):
            Real in${dn(d)} = in[(src_idx * multipoles_per_cell + j) * ${tensor_dim} + ${d}];
            % endfor

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            ${prim.vector_kernels(k_name)}
        }

        % for d in range(tensor_dim):
        atomic_fadd(&out[i * ${tensor_dim} + ${d}], sum${dn(d)});
        % endfor
    }
}
</%def>

<%
kernels = [
    ('one', 1), ('invr', 1), ('tensor_invr', 3), ('laplace_double', 1),
    ('U', 3), ('T', 3), ('A', 3), ('H', 3)
]
%>
% for k, tensor_dim in kernels:
    ${p2p_kernel(k, tensor_dim)}
    ${m2p_kernel(k, tensor_dim)}
% endfor
