import cupy as cp
import cupyx.scipy.fft as cufft
import math


# TODO: Generalize to 3D
def get_structure_factor(grid, real_dens, for_pair_corr=False):
    """
    Calculate the structure factor of a density grid.

    Parameters:
        grid (Grid):
            The grid object.
        real_dens (cparray):
            The density grid.
        for_pair_corr (bool):
            If True, return the structure factor without averaging over shells.

    Returns:
        struct_dists (cparray):
            The distances of the shells.
        s_fact_1d (cparray):
            The structure factor averaged over shells.
        s_fact_2d (cparray):
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
    Calculate the pair correlation function of a density grid.

    Parameters:
        grid (Grid):
            The grid object.
        real_dens (cparray):
            The density grid.

    Returns:
        where_bins (cparray):
            The distances of the shells.
        g_1d (cparray):
            The pair correlation function averaged over shells.
        g_2d (cparray):
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


def get_free_energy(polymer_system, E):
    """
    Function to compute the free energy of the system.

    Parameters:
        E (float):
            The scaled Bjerrum length for the system.

    Returns:
        total_free_energy (float):
            The total free energy of the system.
    """

    polymer_system.update_normal_from_density()
    # initialize free energy array
    free_energy = cp.zeros_like(polymer_system.normal_w[0])
    mu_energy = 0

    # mu squared terms
    for i in range(polymer_system.normal_w.shape[0]):
        #        free_energy += -(1 / (2 * polymer_system.normal_evalues[i])) * cp.square(
        #            polymer_system.normal_w[i]
        #        )
        free_energy -= (
            polymer_system.gamma[i] ** 2 / (2 * polymer_system.normal_evalues[i])
        ) * cp.square(polymer_system.normal_w[i])

    # psi term
    psi_k = cufft.fftn(polymer_system.psi)
    grad_psi_k = psi_k * 1j * polymer_system.grid.k1
    grad_psi = cufft.ifftn(grad_psi_k, s=psi_k.shape)

    if E != 0:
        free_energy += cp.abs(grad_psi) ** 2 / (2 * E)

    total_free_energy = cp.sum(free_energy) * polymer_system.grid.dV

    # Partition energy contribution
    partition_energy = 0.0

    ig_entropy = 0.0
    species_partition = {}
    log_Q = lambda Q: cp.log(Q)
    lin_Q = lambda Q: Q
    for species in polymer_system.Q_dict:

        if polymer_system.ensemble == "canonical":
            can_func = log_Q
        elif polymer_system.ensemble_dict[species] == "C":
            can_func = log_Q
        elif polymer_system.ensemble_dict[species] == "GC":
            can_func = lin_Q
        else:
            raise ValueError(
                "Something went wrong with the dictionary of the ensembles in the energy evaluation"
            )

        if species in polymer_system.poly_dict:
            partition_energy -= (
                polymer_system.poly_dict[species]
                * polymer_system.grid.V
                * can_func(polymer_system.Q_dict[species])
            )
            species_partition[species] = (
                -polymer_system.poly_dict[species]
                * polymer_system.grid.V
                * can_func(polymer_system.Q_dict[species])
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
                * can_func(polymer_system.Q_dict[species])
            )
            species_partition[species] = (
                -polymer_system.solvent_dict[species]
                * polymer_system.grid.V
                * can_func(polymer_system.Q_dict[species])
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
    total_free_energy += partition_energy  # + ig_entropy

    avg_conc = cp.average(
        polymer_system.reduce_phi_all(polymer_system.phi_all),
        axis=range(1, polymer_system.phi_all.ndim),
    )

    # Free energy from homogeneous case (needed for comparing across conditions in
    # gibbs ensemble and others)
    #    total_free_energy += (
    #        (avg_conc @ polymer_system.red_FH_mat @ avg_conc).real * polymer_system.grid.V / 2
    #    )
    return total_free_energy
    # TODO: remove the contributions for the final outcome or institutionalize them


def get_chemical_potential(polymer_system):
    """
    Function to compute the chemical potential of the system.

    Returns:
        chem_pot_dict (dict):
            Dictionary of chemical potentials for each species.
    """

    polymer_system.chem_pot_dict = {}

    avg_mass = cp.sum(polymer_system.phi_all) / polymer_system.grid.k2.size
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
            polymer_system.chem_pot_dict[species] += cp.log(
                polymer_system.Q_dict[species]
            )
            polymer_system.chem_pot_dict[species] += cp.log(
                polymer_system.solvent_dict[species]
            )
            # Enthalpic contribution
            alpha = cp.zeros_like(avg_red_mass)
            alpha[polymer_system.rev_degen_dict[species]] += 1
            # TODO: maybe this should be phi rather than total mass, kind of unclear but I think this is right
            # polymer_system.chem_pot_dict[species] += -(avg_red_mass@polymer_system.red_FH_mat@avg_red_mass.T/2) / polymer_system.N
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


def get_pressure(polymer_system):
    """
    Function to compute the pressure of the system.

    Returns:
        pressure (float):
            Pressure of the system.
    """

    # since the ideal mixture terms are extensive but their underlying
    # functions don't (or weakly, depending on construction) depend on volume we will use their
    # values divided by volume to get their contribution to pressure
    # TODO: functionalize this, it is used multiple times
    ideal_contribution = 0j
    Q_contribution = 0j

    avg_conc = cp.average(
        polymer_system.reduce_phi_all(polymer_system.phi_all),
        axis=range(1, polymer_system.phi_all.ndim),
    )

    ideal_contribution += (avg_conc @ polymer_system.red_FH_mat @ avg_conc).real / 2
    for poly in polymer_system.poly_dict:
        ideal_contribution += polymer_system.poly_dict[poly]
        Q_contribution += polymer_system.dQ_dV_dict[poly]
    for sol in polymer_system.solvent_dict:
        ideal_contribution += polymer_system.solvent_dict[sol]
        Q_contribution += polymer_system.dQ_dV_dict[sol]
    if polymer_system.use_salts:
        polymer_system.get_salt_concs()
        for salt in polymer_system.salts:
            ideal_contribution += polymer_system.salt_concs[salt]
            Q_contribution += polymer_system.dQ_dV_dict[salt]

    # TODO: Add salt
    # in the alternative formulation we only need the partition function terms
    # differs slightly from Villet because we need to solve segment by
    # segment, but we will offload this into the density operator
    pressure = ideal_contribution + Q_contribution

    return pressure
