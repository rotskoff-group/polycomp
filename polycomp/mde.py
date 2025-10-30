from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.fft as cufft

from polycomp._kernels import exp_mult, exp_mult_comp

if TYPE_CHECKING:
    import numpy as np
    from grid import Grid


def s_step(q_r: cp.ndarray, h: float, w_P: cp.ndarray, grid: Grid) -> cp.ndarray:
    """
    Function to run one step of modified diffusion integration using Trotter
    decomposition.

    Form of the integration is

    $q(s + \\Delta s, \\boldsymbol{r}) =
    e^{-\\frac{\\psi(s,\\boldsymbol{r})}{2} \\Delta s}
    e^{\\nabla^2 \\Delta s} e^{-\\frac{\\psi(s,\\boldsymbol{r})}{2} \\Delta s}
    q(s,\\boldsymbol{r})$.

    Uses cupy custom functions to more efficiently compute the products with C
    code, otherwise just products and Fourier transforms.

    Parameters
    ----------
    q_r
        Propagator value for the previous step $q(s,\\boldsymbol{r})$
        (or $q^{\\dagger}(s,\\boldsymbol{r})$).
    h
        Integration step size $\\Delta s$.
    w_P
        Array for the corresponding chemical potential field $\\psi(s,\\boldsymbol{r})$.
    grid
        System grid.

    Returns
    -------

    q_r_t : cp.ndarray
        the propagator $q(s+\\Delta s )$ (or $q^{\\dagger}(s - \\Delta s)$).
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
    struct: np.ndarray,
    h_struct: cp.ndarray,
    species_dict: dict,
    q_r_start: cp.ndarray,
    q_r_dag_start: cp.ndarray,
    grid: Grid,
    fastener=None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Function to integrate one polymer.

    Takes the polymer structure and underlying chemical potential and generates
    values for the modified diffusion equation at every step along the polymer.

    To compute each step this uses the fourth-order Richardson Extrapolation scheme
    where the step is
    $q(s + \\Delta s, \\mathbf{r}) = \\frac{4q_{\\Delta s/2}(s + \\Delta s, \\mathbf{r})
    - q_{\\Delta s}(s + \\Delta s, \\mathbf{r})}{3}$.
    $q_{\\Delta s/2}(s + \\Delta s, \\mathbf{r})$ uses two half width steps and
    $q_{\\Delta s}(s + \\Delta s, \\mathbf{r})$ uses one full width step to compute the
    new values.

    The full integral computes the entire modified diffusion equation for the polymer.
    Will recognize symmetric polymers and efficiently reuse propogators.

    Currently only implemented for linear polymers.

    Parameters
    ----------
    struct
        Monomer ordering along the polymer $m_p(s)$
    h_struct
        CuPy array of floats containing the length of each segment along the polymer
        $\\Delta s (s)$.
    species_dict
        Dictionary of species to ints representing the mapping between monomer types
        and the index for their corresponding density and potential
        $m_p \\rightarrow \\psi_i(\\boldsymbol{r})$.
    q_r_start
        Complex valued start to forward propogator $q(0, \\boldsymbol{r})$.
    q_r_dag_start
        Complex valued start to reverse propogator
        $q^\\dagger(N_P / N, \\boldsymbol{r})$.
    grid
        System grid.

    Returns
    -------

    q_r_s : cp.ndarray
        Complex valued forward propagator along then entire segment chain
        $q(\\boldsymbol{s},\\boldsymbol{r})$
    q_r_dag_s : cp.ndarray
        Complex valued reverse propagator along the entire segment chain
        $q^{\\dagger}(\\boldsymbol{s},\\boldsymbol{r})$
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
