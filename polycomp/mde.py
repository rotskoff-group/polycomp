import cupy as cp
import cupyx.scipy.fft as cufft
from polycomp.kernels import *


def s_step(q_r, h, w_P, grid):
    """
    Function to run one step of Modified diffusion integration.

    Uses cupy custom functions to more efficiently compute the products with C
    code, otherwise just products and Fourier transforms.

    Parameters:
        q_r (cparray):
            sys_dim array of complex128 representing a single q value.
        h (float):
            Integration step size.
        w_P (cparray):
            sys_dim array of complex128 representing effective chemical potential.
        grid (Grid object):
            System grid.
    """

    # Run a single step of Equation 13
    q_k = cp.zeros(q_r.shape, dtype=complex)
    q_r_t = exp_mult(q_r, w_P, h / 2.0)

    q_k = cufft.fftn(q_r_t, s=q_r_t.shape)
    q_k = exp_mult_comp(q_k, grid.k2, h)
    q_r_t = cufft.ifftn(q_k, s=q_k.shape)

    q_r_t = exp_mult(q_r_t, w_P, h / 2.0)
    return q_r_t


def integrate_s(
    struct,
    h_struct,
    species_dict,
    q_r_start,
    q_r_dag_start,
    grid,
    fastener=None,
):
    """
    Function to integrate one polymer.

    Takes the polymer structure and underlying chemical potential and generates
    values for the modified diffusion equation at every step along the polymer.

    Parameters:
        struct (ndarray):
            ndarray of Monomers containing the corresponding monomer type along the
            polymer.
        h_struct (cparray):
            cparray of floats containing the length of each segment along the polymer.
        species_dict (dict):
            Dictionary of species to ints representing the mapping between monomer types
            and the index for their corresponding density and potential.
        q_r_start (cparray):
            cparray of complex128 representing the initial value of q.
        q_r_dag_start (cparray):
            cparray of complex128 representing the initial value of q_dagger.
        grid (Grid object):
            System grid.
    """

    # integrates using each point in struct as an integration point
    # returns the q_r, q_r_dagger, and the key designating point along struct
    # they belong too

    q_r = q_r_start
    q_r_dag = q_r_dag_start

    s_seg = len(struct)

    # write the list of all q_r points along polymer structure
    q_r_s = cp.zeros((s_seg + 1, *q_r.shape), dtype=complex)
    q_r_dag_s = cp.zeros((s_seg + 1, *q_r.shape), dtype=complex)

    # index to ensure sampling happened at the correct places
    i = 0

    # itialize q_r_s at q_r_start
    q_r_s[0] = q_r
    q_r_dag_s[-1] = q_r_dag

    # advance, integrate and write key and q_r_s
    for bead in struct:
        # implements a 4th order integrator
        q_r_1 = s_step(q_r, h_struct[i], species_dict[bead], grid)
        q_r_2 = s_step(q_r, h_struct[i] / 2.0, species_dict[bead], grid)
        q_r_2 = s_step(q_r_2, h_struct[i] / 2.0, species_dict[bead], grid)
        q_r = (4 * q_r_2 - q_r_1) / 3
        i += 1
        q_r_s[i] = q_r

    # If the polymer is symmetric, we can use the symmetry to save time
    if (
        cp.array_equal(struct, cp.flip(struct))
        and cp.array_equal(h_struct, cp.flip(h_struct))
        and fastener is None
    ):
        q_r_dag_s = cp.flip(q_r_s, axis=0)
        return q_r_s, q_r_dag_s

    if fastener is not None:
        q_r_s[0] = q_r = cp.array(fastener.density, dtype=complex) / q_r_dag_s[-1]

    # retreat, integrate, and write q_r_dag_s
    for bead in reversed(struct):
        i -= 1
        q_r_dag_1 = s_step(q_r_dag, h_struct[i], species_dict[bead], grid)
        q_r_dag_2 = s_step(q_r_dag, h_struct[i] / 2.0, species_dict[bead], grid)
        q_r_dag_2 = s_step(q_r_dag_2, h_struct[i] / 2.0, species_dict[bead], grid)
        q_r_dag = (4 * q_r_dag_2 - q_r_dag_1) / 3
        q_r_dag_s[i] = q_r_dag

    return q_r_s, q_r_dag_s
