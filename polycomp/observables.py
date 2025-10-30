from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.fft as cufft

if TYPE_CHECKING:
    from ft_system import PolymerSystem
    from grid import Grid


# TODO: Generalize to 3D
def get_structure_factor(
    grid: Grid, real_dens: cp.ndarray, for_pair_corr: bool = False
) -> tuple:
    """
    Calculate the structure factor of a density grid. Only works for 2D systems at
    present. Uses pseudospectral convolution. 1D structure factors can be
    jagged due to discrete nature of shells.

    Parameters
    ----------

    grid
        The grid object.
    real_dens
        The density grid to compute the structure factor for.
    for_pair_corr
        If True, return the structure factor without averaging over shells.

    Returns
    -------

    struct_dists : cp.ndarray
        The distance associated with the same indexed `s_fact_1d`.

        (Only returned if `for_pair_corr` is `False`)
    s_fact_1d : cp.ndarray
        The structure factor averaged over all points that have the same distance
        stored in `struct_dists`.

        (Only returned if `for_pair_corr` is `False`)
    s_fact_2d : cp.ndarray
        The structure factor without averaging over shells, same shape as k-space array.
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


def pair_correlation(grid: Grid, real_dens: cp.ndarray) -> tuple:
    """
    Calculate the pair correlation function of a density grid, uses the structure
    method and transforms back to real space. Currently only computes autocorrelation.

    Parameters:
        grid
            The grid object.
        real_dens
            The density grid for which the pair correlation will be calculated.

    Returns:
        where_bins : cp.ndarray
            The distance associated with the same indexed `g_1d`.
        g_1d : cp.ndarray
            The structure factor averaged over all points that have the same distance
            stored in `where_bins`.
        g_2d : cp.ndarray
            The pair correlation function without averaging over shells.
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


def get_free_energy(polymer_system: PolymerSystem, E: float) -> float:
    """

    This function calculates the complete free energy, $F$, which includes
    the ideal gas entropy of mixing and the field-theoretic Hamiltonian, $H$.
    The total free energy is given by:

    $$
    F = \\sum_{j} n_j (\\log C_j - 1) + \\langle H[\\{\\mu_i\\}, \\varphi] \\rangle_s
    $$

    where $n_j$ and $C_j$ are the number and concentration of species $j$,
    respectively, and $\\langle H \\rangle_s$ is the spatially-averaged
    Hamiltonian. The Hamiltonian itself is defined as:

    $$
    \\begin{align*}
    H[\\{\\mu_i\\}, \\varphi] = &\\sum_{i=1}^{M} \\frac{\\gamma_i^2}{2B_i}
    from ft_system import PolymerSystem
    \\int_{\\Omega} d\\boldsymbol{r} \\mu_i^2(\\boldsymbol{r}) +
    \\frac{1}{2E} \\int_{\\Omega}
    d\\boldsymbol{r} |\\nabla\\varphi(\\boldsymbol{r})|^2 \\\\
    &- \\sum_{j=1}^{P+S+2} n_j \\log Q_j[\\{\\mu_i\\}, \\varphi]
    + \\frac{V\\boldsymbol{c}^T \\boldsymbol{\\chi} \\boldsymbol{c}}{2}.
    \\end{align*}
    $$

    This only calculates the free energy for a single configuration, so to get a
    proper value, sampling should be done over the free energy.

    Parameters
    ----------
    E
        The scaled Bjerrum length for the system.

    Returns
    -------
    total_free_energy
        The total free energy of the system.
    """

    polymer_system.update_normal_from_density()
    # initialize free energy array
    free_energy = cp.zeros_like(polymer_system.normal_w[0])

    # mu squared terms
    for i in range(polymer_system.normal_w.shape[0]):
        free_energy += -(1 / (2 * polymer_system.normal_evalues[i])) * cp.square(
            polymer_system.normal_w[i]
        )

    # psi term
    psi_k = cufft.fftn(polymer_system.psi)
    grad_psi_k = psi_k * 1j * polymer_system.grid.k1
    grad_psi = cufft.ifftn(grad_psi_k, s=psi_k.shape)

    free_energy += cp.abs(grad_psi) ** 2 / (2 * E)

    total_free_energy = cp.sum(free_energy) * polymer_system.grid.dV

    # Partition energy contribution
    partition_energy = 0.0

    ig_entropy = 0.0
    species_partition = {}
    for species in polymer_system.Q_dict:
        if species in polymer_system.poly_dict:
            partition_energy -= (
                polymer_system.poly_dict[species]
                * polymer_system.grid.V
                * cp.log(polymer_system.Q_dict[species])
            )
            species_partition[species] = (
                -polymer_system.poly_dict[species]
                * polymer_system.grid.V
                * cp.log(polymer_system.Q_dict[species])
            )
            # Ideal gas entropy contribution
            ig_entropy += (
                polymer_system.poly_dict[species]
                * polymer_system.grid.V
                * (cp.log(polymer_system.poly_dict[species]) - 1)
            )
        elif species in polymer_system.solvent_dict:
            partition_energy -= (
                polymer_system.solvent_dict[species]
                * polymer_system.grid.V
                * cp.log(polymer_system.Q_dict[species])
            )
            species_partition[species] = (
                -polymer_system.solvent_dict[species]
                * polymer_system.grid.V
                * cp.log(polymer_system.Q_dict[species])
            )
            # Ideal gas entropy contribution
            ig_entropy += (
                polymer_system.solvent_dict[species]
                * polymer_system.grid.V
                * (cp.log(polymer_system.solvent_dict[species]) - 1)
            )
        elif species in polymer_system.salts:
            salt_conc = polymer_system.salt_concs[species]
            partition_energy -= (
                salt_conc
                * polymer_system.grid.V
                * cp.log(polymer_system.Q_dict[species])
            )
            species_partition[species] = (
                -salt_conc
                * polymer_system.grid.V
                * cp.log(polymer_system.Q_dict[species])
            )
            # Ideal gas entropy contribution
            ig_entropy += salt_conc * polymer_system.grid.V * (cp.log(salt_conc) - 1)
        else:
            print("Bad Species:", species)
            raise ValueError("Couldn't find species in any dictionary")
    total_free_energy += partition_energy + ig_entropy

    avg_conc = cp.average(
        polymer_system.reduce_phi_all(polymer_system.phi_all),
        axis=range(1, polymer_system.phi_all.ndim),
    )

    # Free energy from homogeneous case (needed for comparing across conditions in
    # gibbs ensemble and others)
    total_free_energy += (
        (avg_conc @ polymer_system.red_FH_mat @ avg_conc).real
        * polymer_system.grid.V
        / 2
    )
    return total_free_energy
    # TODO: remove the contributions for the final outcome or institutionalize them


def get_chemical_potential(polymer_system: PolymerSystem) -> None:
    """
    Function to compute the chemical potential of the system.

    This function calculates the chemical potential, $\\mu_j$, for each species $j$
    according to the standard formula in polymer field theory:

    $$
    \\mu_j = \\log C_j - \\log Q_j +
    \\vec{\\kappa}_j^T \\boldsymbol{\\chi} \\boldsymbol{c}
    $$

    where $C_j$ is the concentration, $Q_j$ is the partition function,
    $\\vec{\\kappa}_j$ is the composition vector of species $j$,
    $\\boldsymbol{\\chi}$ is the
    Flory-Huggins interaction matrix, and $\\boldsymbol{c}$ is the vector of average
    monomer concentrations.

    Notes
    -----
    This function modifies the `polymer_system` object in-place by setting
    the `chem_pot_dict` attribute.

    Parameters
    ----------
    polymer_system : PolymerSystem
        The system for which to calculate the chemical potentials. This object
        will be modified.

    """

    polymer_system.chem_pot_dict = {}

    cp.sum(polymer_system.phi_all) / polymer_system.grid.k2.size
    avg_red_mass = polymer_system.remove_degeneracy(
        cp.sum(polymer_system.phi_all, axis=(range(1, polymer_system.phi_all.ndim)))
        / polymer_system.grid.k2.size
    )
    if polymer_system.use_salts:
        polymer_system.get_salt_concs()
    for species in polymer_system.Q_dict:
        polymer_system.chem_pot_dict[species] = 0j
        if species in polymer_system.poly_dict:
            # simulation contribution
            polymer_system.chem_pot_dict[species] -= cp.log(
                polymer_system.Q_dict[species]
            )
            polymer_system.chem_pot_dict[species] += cp.log(
                polymer_system.poly_dict[species]
            )
            alpha = cp.zeros_like(avg_red_mass)

            for h, spec in zip(species.h_struct, species.struct):
                alpha[polymer_system.rev_degen_dict[spec]] += h
            polymer_system.chem_pot_dict[species] += (
                alpha @ polymer_system.red_FH_mat @ avg_red_mass.T
            )

        elif species in polymer_system.solvent_dict:
            # simulation contribution
            polymer_system.chem_pot_dict[species] -= cp.log(
                polymer_system.Q_dict[species]
            )
            polymer_system.chem_pot_dict[species] += cp.log(
                polymer_system.solvent_dict[species]
            )
            # Enthalpic contribution
            alpha = cp.zeros_like(avg_red_mass)
            alpha[polymer_system.rev_degen_dict[species]] += 1
            # TODO: maybe this should be phi rather than total mass,
            # kind of unclear but I think this is right
            # polymer_system.chem_pot_dict[species] += (
            # -(avg_red_mass@polymer_system.red_FH_mat@avg_red_mass.T/2)
            # / polymer_system.N)
            polymer_system.chem_pot_dict[species] += (
                2
                / 2
                * (alpha @ polymer_system.red_FH_mat @ avg_red_mass.T)
                / polymer_system.N
            )

        elif species in polymer_system.salts:
            polymer_system.chem_pot_dict[species] -= cp.log(
                polymer_system.Q_dict[species]
            )
            polymer_system.chem_pot_dict[species] += cp.log(
                polymer_system.salt_concs[species]
            )
        else:
            print("Bad Species:", species)


def get_pressure(polymer_system) -> complex:
    """
    Function to compute the pressure of the system. Pressure is computed as

    $\\begin{align*}
    \\beta\\Pi &= \\sum_{j}^{P+S+2} C_j - \\frac{\\mathbf{c}^T \\boldsymbol{\\chi}
                                                 \\mathbf{c}}{2} \\\\
    &+ \\sum_{j}^{P} \\frac{1}{V} \\int_{\\Omega} d\\mathbf{r}
    \\int_{0}^{\\frac{N_j}{N}} ds \\left[ \\frac{2}{d}
    \\rho_{\\nabla j}(\\mathbf{r},s) + \\rho_j(\\mathbf{r},s)
    \\left( (\\Gamma_2 - \\frac{1}{2}\\Gamma) * (\\psi_{(l/j)}
    - Z_{(l/j)}\\varphi)(\\mathbf{r},s) + (\\Gamma_2 - \\frac{1}{2d}\\Gamma)
    * Z_j \\varphi(\\mathbf{r},s) \\right) \\right] \\\\
    &+ \\sum_{j}^{S+2} \\frac{1}{NV} \\int_{\\Omega} d\\mathbf{r} \\rho_j(\\mathbf{r})
    \\times \\left( (\\Gamma_2 - \\frac{1}{2}\\Gamma)
    * (\\psi_{(l/j)} - Z_{(l/j)}\\varphi)(\\mathbf{r})
    + (\\Gamma_2 - \\frac{1}{2d}\\Gamma) * Z_j \\varphi(\\mathbf{r}) \\right).
    \\end{align*}$

    For this function to operate properly, you **MUST** call
    `PolySystem.get_densities()`
    with `for_pressure=True` before calling this function. It is advised that you call
    your integrator with the `for_pressure` flag set to avoid unnecessary computations
    if you are computing pressure live during sampling or optimization.

    Returns
    -------
    pressure
        Pressure of the system.
    """

    # since the ideal mixture terms are extensive but their underlying
    # functions don't (or weakly, depending on construction) depend on volume
    # we will use their values divided by volume to get their contribution to pressure
    # TODO: functionalize this, it is used multiple times
    ideal_contribution = 0j
    Q_contribution = 0j

    avg_conc = cp.average(
        polymer_system.reduce_phi_all(polymer_system.phi_all),
        axis=range(1, polymer_system.phi_all.ndim),
    )
    #    print(avg_conc)
    homo_contribution = (avg_conc @ polymer_system.red_FH_mat @ avg_conc).real / 2

    for poly in polymer_system.poly_dict:
        ideal_contribution += polymer_system.poly_dict[poly]
        if polymer_system.poly_dict[poly] > 0:
            Q_contribution += polymer_system.dQ_dV_dict[poly]
    for sol in polymer_system.solvent_dict:
        ideal_contribution += polymer_system.solvent_dict[sol]
        if polymer_system.solvent_dict[sol] > 0:
            Q_contribution += polymer_system.dQ_dV_dict[sol]
    if polymer_system.use_salts:
        polymer_system.get_salt_concs()
        for salt in polymer_system.salts:
            ideal_contribution += polymer_system.salt_concs[salt]
            Q_contribution += polymer_system.dQ_dV_dict[salt]

    #    print("Ideal gas: ", ideal_contribution)
    #    print("Homogeneous: ", homo_contribution)
    #    print("Q_change: ", Q_contribution)
    pressure = ideal_contribution + homo_contribution + Q_contribution

    return pressure
