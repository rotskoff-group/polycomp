import cupy as cp
import cupyx.scipy.fft as cufft
import math


# TODO: Generalize to 3D
def get_structure_factor(grid, real_dens, for_pair_corr=False):
    """
    Calculate the structure factor of a density grid.

    Parameters
    ----------
    grid : Grid
        The grid object
    real_dens : cparray
        The density grid
    for_pair_corr : bool
        If True, return the structure factor without averaging over shells

    Returns
    -------
    struct_dists : cparray
        The distances of the shells.
    s_fact_1d : cparray
        The structure factor averaged over shells.
    s_fact_2d : cparray
        The structure factor without averaging over shells.
    """

    # Normalize Densities
    work_dens = real_dens.real / cp.average(real_dens.real)
    s_fact_2d = cp.zeros_like(work_dens)
    wheres = cp.unique(grid.k2)[1::]
    s_fact_1d = cp.zeros(wheres.size)

    # Calculate the structure factor
    phi_k = cufft.fftn(work_dens, s=work_dens.shape)
    s_fact_2d = cp.abs(phi_k) ** 2 * cp.average(real_dens.real) + 1
    if for_pair_corr is True:
        return s_fact_2d

    struct_dists = wheres ** (0.5)

    # Average over shells
    # TODO: Allow for user set binning
    num_bins = wheres.size // 5
    bins = cp.linspace(cp.amin(wheres), cp.amax(wheres), num=num_bins + 1)
    where_bins = bins[:-1] + (bins[1] - bins[0]) / 2
    locations = cp.digitize(grid.k2, bins)
    s_fact_1d = cp.zeros(where_bins.size)

    for i in range(where_bins.size):
        s_fact_1d[i] = cp.average(s_fact_2d[locations == i])
    struct_dists = where_bins ** (0.5)

    return struct_dists, s_fact_1d, s_fact_2d


def pair_correlation(grid, real_dens):
    """
    Calculate the pair correlation function of a density grid

    Parameters
    ----------
    grid : Grid
        The grid object
    real_dens : cparray
        The density grid

    Returns
    -------
    where_bins : cparray
        The distances of the shells
    g_1d : cparray
        The pair correlation function averaged over shells
    g_2d : cparray
        The pair correlation function without averaging over shells
    """

    s2d = get_structure_factor(grid, real_dens, for_pair_corr=True)
    rho = cp.average(real_dens)
    disps = cp.sqrt(cp.sum(grid.grid**2, axis=0))

    g_2d = cufft.ifftn((s2d - 1), s=s2d.shape).real / (rho * real_dens.size)

    wheres = cp.unique(disps)[1::]

    # TODO: Allow for user set binning
    num_bins = wheres.size // 800
    bins = cp.linspace(cp.amin(wheres), cp.amax(wheres), num=num_bins + 1)
    where_bins = bins[:-1] + (bins[1] - bins[0]) / 2
    locations = cp.digitize(disps, bins)
    g_1d = cp.zeros(where_bins.size)

    for i in range(where_bins.size):
        g_1d[i] = cp.average(g_2d[locations == i])

    return where_bins, g_1d, g_2d
