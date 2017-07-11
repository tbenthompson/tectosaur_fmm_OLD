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
void atomic_fadd(volatile __global Real *addr, Real val) {
    union {
        uint u;
        Real f;
    } next, expected, current;
    current.f = *addr;
    do {
        expected.f = current.f;
        next.f = expected.f + val;
        current.u = atomic_cmpxchg(
            (volatile __global uint *)addr, expected.u, next.u
        );
    } while(current.u != expected.u);
}

<%def name="p2p_kernel(K)">
__kernel
void p2p_kernel_${K.name}(__global Real* out, __global Real* in,
        int n_blocks,
        __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_start, __global int* src_n_end,
        __global Real* obs_pts, __global Real* obs_ns,
        __global Real* src_pts, __global Real* src_ns, __global Real* params)
{
    const int block_idx = get_global_id(0); 
    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];
    int src_start = src_n_start[block_idx];
    int src_end = src_n_end[block_idx];

    ${K.constants_code}

    for (int i = obs_start; i < obs_end; i++) {
        % for d in range(3):
            Real x${dn(d)} = obs_pts[i * 3 + ${d}];
            Real nobs${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor

        % for d in range(K.tensor_dim):
        Real kahansum${dn(d)} = 0.0;
        Real kahantemp${dn(d)} = 0.0;
        % endfor

        for (int j = src_start; j < src_end; j++) {
            % for d in range(3):
                Real y${dn(d)} = src_pts[j * 3 + ${d}];
                Real nsrc${dn(d)} = src_ns[j * 3 + ${d}];
            % endfor

            % for d in range(K.tensor_dim):
            Real in${dn(d)} = in[j * ${K.tensor_dim} + ${d}];
            % endfor

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            if (r2 == 0) {
                continue;
            }

            Real sumx = 0.0;
            Real sumy = 0.0;
            Real sumz = 0.0;
            ${K.vector_code}
            % for d in range(K.tensor_dim):
                { //TODO: Is kahan summation necessary here?
                    Real y = sum${dn(d)} - kahantemp${dn(d)};
                    Real t = kahansum${dn(d)} + y;
                    kahantemp${dn(d)} = (t - kahansum${dn(d)}) - y;
                    kahansum${dn(d)} = t;
                }
            % endfor
        }

        % for d in range(K.tensor_dim):
        atomic_fadd(&out[i * ${K.tensor_dim} + ${d}], kahansum${dn(d)});
        % endfor
    }
}
</%def>

<%def name="m2p_kernel(K)">
__kernel
void m2p_kernel_${K.name}(__global Real* out, __global Real* in,
        int n_blocks, __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_idx, int multipoles_per_cell, __global Real* surf,
        __global Real* src_n_center, __global Real* src_n_width, Real inner_r,
        __global Real* obs_pts, __global Real* obs_ns, __global Real* params)
{
    const int block_idx = get_global_id(0);
    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];

    int src_idx = src_n_idx[block_idx];

    Real surf_radius = src_n_width[src_idx] * inner_r * sqrt(3.0);
    % for d in range(3):
    Real center${dn(d)} = src_n_center[src_idx * 3 + ${d}];
    % endfor

    ${K.constants_code}

    for (int i = obs_start; i < obs_end; i++) {
        % for d in range(3):
            Real x${dn(d)} = obs_pts[i * 3 + ${d}];
            Real nobs${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor

        % for d in range(K.tensor_dim):
        Real sum${dn(d)} = 0.0;
        % endfor

        for (int j = 0; j < multipoles_per_cell; j++) {
            % for d in range(3):
                Real nsrc${dn(d)} = surf[j * 3 + ${d}];
                Real y${dn(d)} = surf_radius * nsrc${dn(d)} + center${dn(d)};
            % endfor

            % for d in range(K.tensor_dim):
            Real in${dn(d)} = in[(src_idx * multipoles_per_cell + j) * ${K.tensor_dim} + ${d}];
            % endfor

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            if (r2 == 0) {
                continue;
            }

            ${K.vector_code}
        }

        % for d in range(K.tensor_dim):
        atomic_fadd(&out[i * ${K.tensor_dim} + ${d}], sum${dn(d)});
        % endfor
    }
}
</%def>

<%def name="p2m_kernel(K)">
__kernel
void p2m_kernel_${K.name}(__global Real* out, __global Real* in,
        int n_blocks, __global int* src_n_start, __global int* src_n_end,
        __global int* src_n_idx, int multipoles_per_cell, __global Real* surf,
        __global Real* src_n_center, __global Real* src_n_width, Real outer_r,
        __global Real* src_pts, __global Real* src_ns, __global Real* params)
{
    const int block_idx = get_global_id(0);
    int src_start = src_n_start[block_idx];
    int src_end = src_n_end[block_idx];
    int src_idx = src_n_idx[block_idx];

    Real surf_radius = src_n_width[src_idx] * inner_r * sqrt(3.0);
    % for d in range(3):
    Real center${dn(d)} = src_n_center[src_idx * 3 + ${d}];
    % endfor

    ${K.constants_code}

    for (int i = obs_start; i < obs_end; i++) {
        % for d in range(3):
            Real x${dn(d)} = obs_pts[i * 3 + ${d}];
            Real nobs${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor

        % for d in range(K.tensor_dim):
        Real sum${dn(d)} = 0.0;
        % endfor

        for (int j = 0; j < multipoles_per_cell; j++) {
            % for d in range(3):
                Real nsrc${dn(d)} = surf[j * 3 + ${d}];
                Real y${dn(d)} = surf_radius * nsrc${dn(d)} + center${dn(d)};
            % endfor

            % for d in range(K.tensor_dim):
            Real in${dn(d)} = in[(src_idx * multipoles_per_cell + j) * ${K.tensor_dim} + ${d}];
            % endfor

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            if (r2 == 0) {
                continue;
            }

            ${K.vector_code}
        }

        % for d in range(K.tensor_dim):
        atomic_fadd(&out[i * ${K.tensor_dim} + ${d}], sum${dn(d)});
        % endfor
    }
}
</%def>

% for K in fmm_kernels:
    ${p2p_kernel(K)}
    ${m2p_kernel(K)}
% endfor
