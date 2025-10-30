from __future__ import annotations

import math
import numbers
import warnings
from typing import Union

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np

from polycomp._kernels import kernel_mult_complex, kernel_mult_float
from polycomp.base import Brush, Monomer, Nanoparticle, Polymer
from polycomp.complex_langevin_ETD import CL_RK2
from polycomp.grid import Grid
from polycomp.mde import integrate_s, s_step

__all__ = [
    # From base.py
    "Monomer",
    "Polymer",
    "Brush",
    "Nanoparticle",
    # From grid.py
    "Grid",
    # From mde.py
    "s_step",
    "integrate_s",
    # From complex_langevin_ETD.py
    "CL_RK2",
    # Class defined in THIS file
    "PolymerSystem",
]


class PolymerSystem:
    """
    Base class for the overall method. Includes function for computing density and
    the system state is stored in this object.

    Usage notes:

    N will be set by default to the longest length scale of the system,
    which may be undesirable depending on how other parameters are scaled.


    Attributes
    ----------

    n_species : int
        Number of monomer species in simulation $M$.
    integration_width : float
        Maximum integration width along the polymer $\\Delta s$.
    FH_dict : dict
        Dict of `frozenset(Monomer)` $\\rightarrow$ `float` representing the interaction
        potentials between all possible pairs of monomers.
    polymers : tuple
        List of Polymer objects representing all polymers in the system.
        $\\{P\\}$.
    poly_dict : dict
        Dict of Polymer objects representing the amount of each polymer.
        in solution $\\{p_i \\rightarrow C_{p_i}\\}$.
    solvent_dict : dict
        Dict of Monomer objects representing the amount of each solvent.
        species in solution $\\{S_i \\rightarrow C_{S_i}\\}$.
    Q_dict : dict
        Stores the partition functions for each polymer and monomer during the
        density collection step $\\{P_i/S_i \\rightarrow Q_j\\}$.
    dQ_dV_dict : dict
        Stores dQ/dV values of each polymer and monomer for pressure
        calculations $\\{P_i/S_i \\rightarrow \\frac{\\partial Q_j}{\\partial V}\\}$.
    FH_matrix : cp.ndarray
        Flory-Huggins matrix representing the interaction potentials between all
        species $\\boldsymbol{\\chi} N$.
    grid : Grid
        Grid object for the simulation.
    w_all : cp.ndarray
        Stacked, complex arrays  representing the chemical potential field for
        each monomer species at every grid point in the real density
        representation.  $\\boldsymbol{b}\\boldsymbol{\\mu}(\\boldsymbol{r})$
    normal_w : cp.ndarray
        Stacked, complex arrays representing the chemical potential field at
        each grid point for fields in the normal mode representation.
        $\\{\\mu_i(\\boldsymbol{r})\\}$
    w_all_smeared : cp.ndarray
        Stacked, smeared, complex arrays created when we need to change the basis out
        of `normal_w`, but also apply smearing functions (especially if $\\alpha_i$
        are not all the same).
    psi : cp.ndarray
        Complex array representing the electrostatic potential at each grid.
        point $\\varphi(\\boldsymbol{r})$.
    smear_arr : cp.ndarray
        Array of smearing constant for the simulation where the index corresponds to the
        same indexed field $\\{\\alpha_i\\}$. Feature is still experimental, has not
        been rigorously tested for non-uniform arrays. If array is uniform, system has
        a single smearing constant.
    psi_smear : float
        Smearing constant for electric charge $\\alpha_{\\varphi}$.
    monomers : tuple
        All monomer types in system with fixed ordering for proper indexing $\\{m\\}$.
    red_FH_mat : cp.ndarray
        FH matrix with degenerate modes removed.
    degen_dict : dict
        Dictionary mapping the identical species in the non-degenerate
        representation to their equivalents in the degenerate representation
        $\\{\\{m\\}_{degen} \\rightarrow i\\}$.
    rev_degen_dict : dict
        Dict listing the indices of species in the non-degenerate representation
        given their degenerate representation $\\{i \\rightarrow \\{m\\}_{degen}\\}$.
    normal_evalues : cp.ndarray
        CuPy Array of floats representing the eigenvalues of the normal mode
        decomposition $\\{B_i\\}$.
    normal_modes : cp.ndaray
        Same as normal modes. Plan to deprecate.
    A_ij : cp.ndarray
        CuPy Array of floats representing the matrix of eigenvalues of the normal
        mode decomposition, notated $\\boldsymbol{b}$ (sometimes $A_{ij}$).
    A_inv : cp.ndarray
        Inverse of A_ij, notated $\\boldsymbol{b}^{-1}$.
    gamma : cp.ndarray
        Complex vector indexed as normal_evalues, entries are 1 eigenvalues is less
        than 0 and $i$ for eigenvalues greater than 0, notaed $\\gamma_i$.
    phi_all : cp.ndarray
        Complex density of each monomer species at every grid point in the real density
        representation, uses standard monomer indexing, notated $\\{\\rho_i\\}$.
    phi_salt : cp.ndarray
        Density of each salt species at every grid point in the real density. Only
        active when using volume-less, non-FH interacting salts.
    salts : tuple or None
        Tuple containing salt `Monomer` objects, if they exist
    salt_concs : dict
        A dictionary mapping salt `Monomer` objects to their calculated concentrations.
        Updated by the `get_salt_concs` method to automatically neutralize net charge.
    salt_pos : Monomer or None
        Generic, non-FH interacting positive salt monomer, if it exists
    salt_neg : Monomer or None
        Generic, non-FH interacting negative salt monomer, if it exists
    has_nanps : bool
        True if there are nanoparticles in the system.
    nanps : Tuple or None
        List of nanoparticle species if they exist
    N : float
        Characteristic length $N$ of the system. By default set to the longest polymer
        in the system.
        Choice is material, if two systems are to be directly compared this
        value needs to be the same between them or many corrections need to be made.
        Rescales entire system.
    use_salts : bool
        True if there are volume-less, non-FH salts in the system.
    ordered_spec : tuple
        Tuple of species in the order they are stored in the density matrix.
    c_s : float
        Total salt concentration.
    chem_pot_dict : dict, optional
        A dictionary mapping each species object to its calculated chemical
        potential. This attribute is only
        set by the `polycomp.observables.get_chemical_potential` function.
        Default is not present.




    Parameters
    ----------

    monomers
        List of monomer species $\\{m\\}$. System will generally reorder them to put the
        solvents last, but interactions will be correct.
    polymers
        List of polymer species $\\{P\\}$.
    spec_dict
        Dict listing the amount of each polymer and solvent species in solution
        $\\{p_i/S_i \\rightarrow C_{i}\\}$.
    FH_dict
        Dict corresponding a frozenset for each pair of monomers and their
        FH interaction term.
    grid
        Grid object for the simulation.
    smear
        Gaussian smearing constant, or array of constants $\\alpha$ or $\\{\\alpha\\}$.
        Arrays correspond to different species, but this method is still experimental.
    psi_smear
        If provided gives the electrostatic smearing constant $\\alpha_{\\varphi}$
    salt_conc
        Total salt concentration, default 0.0. Salts added here will not interact except
        via the charge field and will automatically balance the charge of the system.
    integration_width
        Maximum integration width $\\Delta s$, default 0.2. Defined as fraction of $N$.
    custom_salts
        List of salt species, default None leads to salts with charge $\\pm1$. Still
        experimental.
    nanoparticles
        List of nanoparticle species, default None.

    Raises
    ------
    ValueError
        Raised if there is a species that is not a polymer or monomer in
        the species dictionary.
    """

    def __init__(
        self,
        monomers: list,
        polymers: list,
        spec_dict: dict,
        FH_dict: dict,
        grid: Grid,
        smear: Union[float, cp.ndarray],
        psi_smear: float = 0,
        salt_conc: float = 0.0,
        integration_width: float = 0.2,
        N: float = None,
        custom_salts: tuple = None,
        nanoparticles: tuple = None,
    ) -> None:

        super(PolymerSystem, self).__init__()

        # add grid
        self.grid = grid
        self.n_species = len(monomers)
        self.FH_dict = FH_dict

        # we want the last monomer species to be solvent if possible
        self.set_monomer_order(monomers)

        # sort the species dictionary into polymer and solvent components
        self.polymers = polymers
        self.poly_dict = {}
        self.solvent_dict = {}
        self.Q_dict = {}

        # Set up nanoparticles if needed
        self.has_nanps = False
        if nanoparticles is not None:
            self.nanps = nanoparticles
            self.has_nanps = True

        # Makes sure everything is a polymer or a monomer and checks the total
        # density
        check_frac = 0
        for spec in spec_dict.keys():
            check_frac += spec_dict[spec]
            if spec.__class__.__name__ == "Polymer":
                self.poly_dict[spec] = spec_dict[spec]
            elif spec.__class__.__name__ == "Monomer":
                self.solvent_dict[spec] = spec_dict[spec]
            else:
                raise ValueError("Unknown member of species dictionary")

        # The longest species in the mix is designated as having length of N by default
        if N is not None:
            self.N = N
        elif self.poly_dict:
            self.N = max([x.total_length for x in self.poly_dict.keys()])
        else:
            self.N = 1
        self.integration_width = integration_width

        for poly in self.poly_dict.keys():
            poly.total_length = poly.total_length

        # build flory huggins matrix
        self.FH_matrix = cp.zeros((self.n_species, self.n_species))

        # THIS FIX IS FULLY CRITICAL
        # PREVIOUSLY WOULD SHUFFLE THE FH MATRIX UNLESS SOLVENTS ARE LAST
        for i in range(len(self.monomers)):
            for j in range(len(self.monomers)):
                self.FH_matrix[i, j] = self.FH_dict[
                    #                    frozenset((monomers[i], monomers[j]))
                    frozenset((self.monomers[i], self.monomers[j]))
                ]

        # write the actual integration frameworks to each polymer
        for polymer in self.poly_dict:
            polymer.build_working_polymer(
                self.integration_width, polymer.total_length / self.N
            )

        # Check for degeneracies in representation and correct if necessary
        self.find_degeneracy()

        # generate FH matrix and find normal modes / eigenvalues
        self.assign_normals()
        self.get_gamma()

        self.A_inv = cp.linalg.inv(self.A_ij)

        size = len(self.normal_evalues)
        hold = cp.zeros((size, size))
        for i in range(size):
            hold[i, i] = self.normal_evalues[i]

        # Initialize all fields (currently to zero)
        self.w_all = cp.zeros(
            [self.red_FH_mat.shape[0]] + list(self.grid.k2.shape),
            dtype=complex,
        )
        self.psi = cp.zeros_like(self.grid.k2, dtype=complex)

        # Initialize mu field
        self.update_normal_from_density()
        if isinstance(smear, numbers.Number):
            self.smear_arr = cp.ones_like(self.red_FH_mat) * smear
            self.psi_smear = smear
        else:
            self.smear_arr = smear
            self.psi_smear = psi_smear

        # set a canonical ordering for the species (helpful for gibbs ensemble)
        canonical_ordering = []
        for species in spec_dict.keys():
            canonical_ordering.append(species)

        if abs(salt_conc) == 0:
            self.use_salts = False
            self.ordered_spec = tuple(canonical_ordering)
            return
        self.use_salts = True
        self.c_s = salt_conc
        if custom_salts is None:
            self.salt_pos = Monomer("salt+", 1, identity="salt", has_volume=False)
            self.salt_neg = Monomer("salt-", -1, identity="salt", has_volume=False)
            self.salts = (self.salt_pos, self.salt_neg)
        else:
            self.salts = custom_salts
        if len(self.salts) not in (0, 2):
            raise NotImplementedError("Unusual number of salts")

        # Fix the ordering
        for salt in self.salts:
            canonical_ordering.append(salt)
        self.ordered_spec = tuple(canonical_ordering)

        return

    def set_monomer_order(self, monomers: list) -> None:
        """
        Permanently affixes the order of the monomers.

        Orders the monomers so that solvents tend to be last and then writes
        them into a tuple.

        Updates the following attributes:

        - `self.monomers`

        Parameters
        ----------

        monomers
            Monomer objects to be ordered.
        """

        forward_count = 0
        reverse_count = len(monomers) - 1
        temp_list = monomers.copy()
        for monomer in monomers:
            if monomer.identity == "solvent":
                temp_list[reverse_count] = monomer
                reverse_count -= 1
            else:
                temp_list[forward_count] = monomer
                forward_count += 1
        self.monomers = tuple(temp_list)
        return

    def find_degeneracy(self) -> None:
        """
        Updates and combines species with identical FH interactions. Singular matrices
        cause problems, so all degenerate species need to be combined for
        the purposes of writing the normal basis. This function
        identifies and combines trivial degenerate species,
        and then creates the dictionaries needed
        to map in and out of the non-degenerate representation.

        The matrix may still be singular after calling this method if there are
        non-trivial degeneracies. Currently such a system cannot be simulated in this
        codebase and this will return an error.

        Updates the following attributes:

        - `self.red_FH_mat`
        - `self.degen_dict`
        - `self.rev_degen_dict`

        Raises
        ------

        ValueError
            If the degeneracy cannot be reduced (usually caused by more complicated
            degeneracies than just identical FH parameters).
        """

        degen_sets = []
        # identify degeneracy
        for i in range(self.FH_matrix.shape[0]):
            for j in range(i + 1, self.FH_matrix.shape[0]):
                if np.allclose(self.FH_matrix[i], self.FH_matrix[j]):
                    degen_sets.append({i, j})
        reducing = True

        # Horrible code to combine the degeneracies
        while reducing:
            reducing = False
            return_to_outer_loop = False
            for i in range(len(degen_sets)):
                if return_to_outer_loop:
                    break
                for j in range(i + 1, len(degen_sets)):
                    if len(degen_sets[i].union(degen_sets[j])) != len(
                        degen_sets[i]
                    ) + len(degen_sets[j]):
                        return_to_outer_loop = True
                        reducing = True
                        degen_sets.append(degen_sets[i].union(degen_sets[j]))
                        degen_sets.pop(i)
                        degen_sets.pop(j)
                        break

        degen_lists = [sorted(x) for x in degen_sets]
        # generate new non-degenerate matrix:
        mask = np.ones(self.FH_matrix.shape[0], bool)
        # generate non-degenerate FH matrix
        for x in degen_lists:
            mask[x[1:]] = 0
        kept_indices = np.arange(len(mask))[mask]
        self.red_FH_mat = self.FH_matrix[kept_indices][:, kept_indices]
        # write a dictionary to record the new indices of the FH matrix to the
        # original species
        self.degen_dict = {}
        self.rev_degen_dict = {}
        for i in range(kept_indices.size):
            modified = False
            for degen in degen_lists:
                if kept_indices[i] in degen:
                    modified = True
                    self.degen_dict[i] = [self.monomers[k] for k in degen]
                    for j in degen:
                        self.rev_degen_dict[self.monomers[j]] = i
            if not modified:
                self.degen_dict[i] = [self.monomers[kept_indices[i]]]
                self.rev_degen_dict[self.monomers[kept_indices[i]]] = i
        if cp.linalg.det(self.red_FH_mat) == 0:
            raise ValueError(
                "The Flory-Huggins matrix is still singular after degeneracy removal"
            )
        return

    def remove_degeneracy(self, array: cp.ndarray) -> cp.ndarray:
        """
        This function sums over degenerate elements.

        This is called external to the function when an external method needs the
        reduced representation and allows access to the non-degenerate version of any
        appropriately shaped array.

        Parameters
        ----------

        array
            Array like w_all to have degenerate elements summed over. Must
            be ordered the same way as w_all.
        """

        fixed_array = cp.zeros(
            [len(self.degen_dict)] + list(array.shape[1:]), dtype=complex
        )

        for i in range(fixed_array.shape[0]):
            for mon in self.degen_dict[i]:
                fixed_array[i] += array[self.monomers.index(mon)]
        return fixed_array

    def assign_normals(self) -> None:
        """
        Assigns the normal eigenvalues and eigenvectors

        Takes a non-degenerate FH matrix and generates the corresponding normal
        mode decomposition factors of A_ij and eigenvalues.

        Updates the following attributes:

        - `self.normal_evalues`
        - `self.normal_modes`
        - `self.A_ij`
        - `self.A_inv`
        """

        # assign coefficients for normal mode tranform
        self.normal_evalues, self.normal_modes = cp.linalg.eigh(self.red_FH_mat)

        idx = self.normal_evalues.argsort()[::-1]
        self.normal_evalues = self.normal_evalues[idx]
        self.normal_modes = self.normal_modes[:, idx]

        warning_thresh = 1e-3 * cp.amax(cp.abs(self.normal_evalues))
        if cp.amin(cp.abs(self.normal_evalues)) <= warning_thresh:
            danger = cp.amin(cp.abs(self.normal_evalues))
            warnings.warn(
                "Minimum eigenvalue is "
                + "{:.3}".format(danger)
                + " which is very small and likely to cause problems"
            )

        self.A_ij = self.normal_modes
        self.A_inv = cp.linalg.inv(self.A_ij)

        return

    def get_gamma(self) -> None:
        """
        Examines eigenvalues and assigns the correct values to gamma based on their
        sign.

        Updates the following attributes:

        - `self.gamma`
        """

        # determine which fields are real and imaginary and assign correct gamma
        gamma = cp.zeros(self.normal_evalues.size, dtype="complex128")
        gamma += 1j * (self.normal_evalues > 0)
        gamma += cp.logical_not(self.normal_evalues > 0)
        self.gamma = gamma
        return

    def randomize_array(self, array: cp.ndarray, noise: float) -> cp.ndarray:
        """
        Generates a random valued array in the same shape as the input array.

        Noise is Gaussian distributed around zero with a variance of `noise`.

        Parameters
        ----------

        array
            Array used to determine the shape of the output noise.
        noise
            Variance of desired noise.

        Returns
        -------

        array_out
            Output array with random noise in the specified configuration.
        """

        array_out = cp.random.normal(0, math.sqrt(noise), size=array.shape) + 0j
        return array_out

    def map_dens_from_norm(self, w_like_array: cp.ndarray) -> cp.ndarray:
        """
        Map any input array from real density to normal representation.

        The first axis of w_like_array must be shaped like w_all's first axis.

        Parameters
        ----------

        w_like_array
            Array in real density space to be transformed.

        Returns
        -------

        new_array
            Array of input in the normal basis.
        """

        new_array = (w_like_array.T @ self.A_ij.T).T
        return new_array

    def update_density_from_normal(self) -> None:
        """
        Updates `self.w_all` to match the current state of `self.normal_w`. If the
        normal basis has been updated and is in the correct state this function should
        be called so the entire system has the correct values. Does not account for
        smearing.

        Updates the following attributes:

        - `self.w_all`
        """
        self.w_all = self.map_dens_from_norm(self.normal_w)
        return

    def update_density_from_normal_smeared(self) -> None:
        """
        Updates `self.w_all` to match the current state of `self.normal_w` accounting
        for the smearing of all different fields. If
        normal basis has been updated and is in the correct state this function should
        be called so the entire system has the correct values. Usually this should be
        called over `update_density_from_normal`, and is mandatory if there are
        different smearing length scales as the smearing must happen before the change
        of basis.

        Updates the following attributes:

        - `self.w_all`
        """
        self.w_all_smeared = cp.zeros_like(self.w_all)
        hold_w_smeared = cp.zeros_like(self.w_all)
        for i in range(hold_w_smeared.shape[0]):
            for j in range(hold_w_smeared.shape[0]):
                hold_w_smeared[j] = self.gaussian_smear(
                    self.normal_w[j], self.smear_arr[i, j]
                )
            hold_w_all = self.map_dens_from_norm(hold_w_smeared)
            self.w_all_smeared[i] = hold_w_all[i]

    def dens_from_norm_generic_smeared(
        self, field_array: cp.ndarray, kernel_array: cp.ndarray
    ) -> cp.ndarray:
        """
        Generic version of `update_density_from_normal_smeared` that can apply any
        smearing kernel to any appropriately shaped array and make the basis transform.
        Currently only used to update the desnities when computing the pressure.

        Parameters
        ----------
        kernel_array
            Array with shape like FH x dims.
        field_array
            Array with shape like `self.w_all`.

        Returns
        -------
        out_array : cp.ndarray
            `field_array` smeared by `kernel_array` and basis transformed.
        """
        hold_out = cp.zeros_like(field_array)
        out_array = cp.zeros_like(field_array)
        for i in range(hold_out.shape[0]):
            for j in range(hold_out.shape[0]):
                hold_out[j] = self.convolve(field_array[j], kernel_array[i, j])
            hold_out = self.map_dens_from_norm(hold_out)
            out_array[i] = hold_out[i]
        return out_array

    def map_norm_from_dens(self, w_like_array: cp.ndarray) -> cp.ndarray:
        """
        Map any array from normal to real density representation.

        The first axis of w_like_array must be shaped like w_all's first axis. Does not
        apply any smearing transform.

        Parameters
        ----------

        w_like_array
            Array in density space to be transformed.

        Returns
        -------

        new_array
            Array in normal space after basis transform.
        """

        new_array = (w_like_array.T @ (self.A_inv.T)).T
        return new_array

    def update_normal_from_density(self) -> None:
        """
        Updates `self.noram_w` to match the current state of `self.w_all`. If the
        density basis has been updated and is in the correct state this function should
        be called so the entire system has the correct values. Does not account for
        smearing.

        Updates the following attributes:

        - `self.normal_w`
        """
        # update the normal mode representation to match current real
        # represenation
        self.normal_w = self.map_norm_from_dens(self.w_all)
        return

    def map_norm_from_dens_smeared(self, w_like_array: cp.ndarray) -> cp.ndarray:
        """
        Maps the input array in the normal basis into the density basis and accounts
        for smearing which may have to occur synchronously with the basis transform.

        Parameters
        ----------

        w_like_array
            Array with the same structure as the normal basis chemical potential field

        Returns
        -------

        smeared_output
            Equivalent array in the density basis with smearing

        """
        smeared_output = cp.zeros_like(w_like_array)
        hold_in_smeared = cp.zeros_like(w_like_array)
        for i in range(hold_in_smeared.shape[0]):
            for j in range(hold_in_smeared.shape[0]):
                # Orientation of self.smear_arr is reversed, which is symmetric so it
                # shouldn't matter but this is formally right
                hold_in_smeared[j] = self.gaussian_smear(
                    w_like_array[j], self.smear_arr[j, i]
                )
            hold_out = self.map_norm_from_dens(hold_in_smeared)
            smeared_output[i] = hold_out[i]
        return smeared_output

    def reduce_phi_all(self, phi_all: cp.ndarray) -> cp.ndarray:
        """
        Converts an array with information about the density of all the species into
        an equivalent array where species with identical FH interactions have been
        summed over.

        Parameters
        ----------

        phi_all
            Array like `phi_all` to be reduced

        Returns
        -------
        red_phi_all : cp.ndarray
            Equivalent array with degenerate density combined and indexing matching
            `self.degen_dict` or `self.rev_degen_dict`
        """

        red_phi_all = cp.zeros(
            [len(self.degen_dict)] + list(phi_all.shape[1:]), dtype=complex
        )
        for i in range(red_phi_all.shape[0]):
            for mon in self.degen_dict[i]:
                red_phi_all[i] += phi_all[self.monomers.index(mon)]
        return red_phi_all

    def convolve(self, array: cp.ndarray, kernel_k: cp.ndarray) -> cp.ndarray:
        """
        Convolve any array with a given kernel.

        The kernel is in k-space, and the array is in real space, uses a cupy C kernel
        for performance.

        Parameters
        ----------

        array
            Real space array to be convolved.
        kernel_k
            Kernel of the same last dimensions in k-space.

        Returns
        -------

        conv
            Convolution of the array and kernel.
        """

        # standard FFT
        array_k = cufft.fftn(array, s=array.shape)

        # More efficient convolution kernels than default cp
        if kernel_k.dtype == "float64":
            kernel_mult = kernel_mult_float
        elif kernel_k.dtype == "complex128":
            kernel_mult = kernel_mult_complex

        # multiply kernel with array
        conv_k = kernel_mult(array_k, kernel_k)

        # Inverse transform
        conv = cufft.ifftn(conv_k, s=array.shape)

        return conv

    def gaussian_smear(self, array: cp.ndarray, alpha: float) -> cp.ndarray:
        """
        Smear an array by a Gaussian pseudospectrally.

        Parameters
        ----------

        array
            Array to be smeared.
        alpha
            Variance of Gaussian to be smeared by.

        Returns
        -------

        array_r
            Smeared array
        """

        # generate convolution kernel
        gauss_k = cp.exp(-self.grid.k2 * alpha**2 / 2)

        # convolve
        array_r = self.convolve(array, gauss_k)
        return array_r

    def laplacian(self, array: cp.ndarray) -> cp.ndarray:
        """
        Compute the laplacian or a spatial array on the grid pseudospectrally.

        Parameters
        ----------

        array
            Array for which the laplician will be taken

        Returns
        -------
        lap_array
            Laplacian of the input array
        """

        # internal gradient for species on the grid
        lap_array = self.convolve(array, -self.grid.k2)
        return lap_array

    def get_net_saltless_charge(self) -> float:
        """
        Get the net charge of the system without non-FH salt species.

        Returns
        -------

        total_charge
            Net charge of the system
        """

        total_charge = 0
        for poly in self.polymers:
            charge_struct = (
                cp.array([bead.charge for bead in poly.struct]) * poly.h_struct
            )
            total_charge += cp.sum(charge_struct) * self.poly_dict[poly] * self.grid.V
        return total_charge

    def get_total_charge(self, include_salt: bool = True) -> cp.ndarray:
        """
        Get the total charge of the system at every point in space.

        Parameters
        ----------
        include_salt
            Whether to include salt in the total charge

        Returns
        -------

        total_charge
            Array of net charge of the system as a function of space
        """

        total_charge = cp.zeros_like(self.psi)
        for i in range(len(self.monomers)):
            total_charge += self.monomers[i].charge * self.phi_all[i]

        # Break out now if no salts
        if not include_salt or not self.use_salts:
            return total_charge

        for i in range(len(self.salts)):
            total_charge += self.salts[i].charge * self.phi_salt[i]
        return total_charge

    def get_densities(self, for_pressure: bool = False) -> None:
        """
        Function to get the densities from a given set of potentials.

        Uses all of the configurations in the polysystem to determine the
        integration scheme. By default will compute for a continuous gaussian chain
        with an effective trapezoid integrator, can be configured for a discrete chain.
        Expensive function, calls should be minimized. Pressure increases expense.

        Updates the following attributes:

        - `self.phi_all`
        - `self.dQ_dV_dict`
        - `self.Q_dict`

        Parameters
        ----------

        for_pressure
            Whether there is a pressure calculation planned before another call of this
            function.

        Raises
        ------

        ValueError
            Raised if $Q_p(s)$ is not the same for all points along the polymer. This
            is usually the case because either there is something wrong with
            the integration plan or the fields have gone unstable and achieved
            unphysical values. Constraint has been relaxed somewhat because structures
            can transiently violate this rule.
        """

        q_r0 = cp.ones_like(self.w_all[0])
        q_r_dag0 = cp.ones_like(self.w_all[0])
        self.phi_all = cp.zeros(
            [len(self.rev_degen_dict)] + list(self.grid.k2.shape),
            dtype=complex,
        )

        # build P_species from w_all and poly_list
        P_species = {}
        if for_pressure:
            self.dQ_dV_dict = {}
            self.dQ_dV_dict.clear()
            P_press_species = {}
            gauss_12_arr = -cp.exp(
                -self.grid.k2
                * self.smear_arr[..., *(None,) * self.grid.k2.ndim] ** 2
                / 2
            ) * (
                self.grid.k2
                * self.smear_arr[..., *(None,) * self.grid.k2.ndim] ** 2
                / self.grid.ndims
                - 1 / 2
            )

            gauss_16 = -cp.exp(-self.grid.k2 * self.psi_smear**2 / 2) * (
                self.grid.k2 * self.psi_smear**2 / self.grid.ndims
                - 1 / (2 * self.grid.ndims)
            )

        self.update_density_from_normal_smeared()
        self.update_density_from_normal()

        # Sometimes the value of the partition function can exceed the floating point
        # limit, in those cases this will try to fix the problem automatically, but
        # it is experimental

        try:
            if self.big_number_safe:
                self.w_all -= (
                    cp.amin(
                        self.w_all, axis=tuple(range(1, self.w_all.ndim)), keepdims=True
                    )
                    / 2
                )
        except AttributeError:
            pass

        for monomer in self.monomers:
            if monomer.has_volume:
                # effective field from total of potentials
                P_species[monomer] = self.w_all_smeared[
                    self.rev_degen_dict[monomer]
                ] + self.gaussian_smear(
                    self.psi * monomer.charge,
                    self.psi_smear,
                )

                # This is the derivative smeared fields for each monomer type
                if for_pressure:
                    P_press_species[monomer] = (
                        self.dens_from_norm_generic_smeared(
                            self.normal_w, gauss_12_arr
                        )[self.rev_degen_dict[monomer]]
                    ) + self.convolve(self.psi * monomer.charge, gauss_16)
        hold_phi_del_part = 0j
        hold_dens_part = 0j
        # Iterate over all polymer types
        for polymer in self.poly_dict:
            f_poly = self.poly_dict[polymer]
            if f_poly == 0:
                continue

            # step size along polymer
            if polymer.fastener is None:
                q_r_s, q_r_dag_s = integrate_s(
                    polymer.struct,
                    polymer.h_struct,
                    P_species,
                    q_r0,
                    q_r_dag0,
                    self.grid,
                )
            else:
                q_r_s, q_r_dag_s = integrate_s(
                    polymer.struct,
                    polymer.h_struct,
                    P_species,
                    q_r0,
                    q_r_dag0,
                    self.grid,
                    fastener=polymer.fastener,
                )
            # Partition function as a function of s
            Q_c = q_r_dag_s * q_r_s
            Q_c = self.reindex_Q_c(Q_c)

            # partition function across entire polymer
            Q = (
                cp.average(cp.sum(Q_c, axis=tuple(range(1, Q_c.ndim))))
                * self.grid.dV
                / self.grid.V
            )
            Q = cp.sum((Q_c)[0]) * self.grid.dV / self.grid.V

            if for_pressure:
                lap_q_r_s = cp.zeros_like(q_r_s)
                for i in range(lap_q_r_s.shape[0]):
                    lap_q_r_s[i] = self.laplacian(q_r_s[i])

                Q_del_c = q_r_dag_s * lap_q_r_s
                Q_del_c = self.reindex_Q_c(Q_del_c)

                phi_del = cp.sum((Q_del_c.T * polymer.h_struct).T, axis=0) / Q

                self.dQ_dV_dict[polymer] = (
                    -f_poly
                    * 2
                    / self.grid.ndims
                    * cp.sum(phi_del)
                    * self.grid.dV
                    / self.grid.V
                )
                hold_phi_del_part += (
                    -f_poly
                    * 2
                    / self.grid.ndims
                    * cp.sum(phi_del)
                    * self.grid.dV
                    / self.grid.V
                )

            # check that Q is equal across integral (necessary condition for
            # equilibrium to be valid)
            if not cp.allclose(
                cp.sum(Q_c, axis=tuple(range(1, len(Q_c.shape))))
                * self.grid.dV
                / self.grid.V,
                Q,
                rtol=1e-4,
            ):
                print(cp.sum(Q_c, axis=tuple(range(1, len(Q_c.shape)))))
                raise ValueError("Q_c not equal across integral")
            #            print(cp.sum(Q_c, axis=tuple(range(1, len(Q_c.shape)))))

            self.Q_dict[polymer] = cp.copy(Q)

            # generate phi's by summing over partition function in correct areas
            for i in range(len(self.monomers)):
                self.phi_all[i] += (
                    cp.sum(
                        (Q_c.T * polymer.h_struct).T[
                            polymer.struct == self.monomers[i]
                        ],
                        axis=0,
                    )
                    * f_poly
                    / (Q)
                )
                if for_pressure:
                    self.dQ_dV_dict[polymer] += (
                        cp.sum(
                            cp.sum(
                                (Q_c.T * polymer.h_struct).T[
                                    polymer.struct == self.monomers[i]
                                ],
                                axis=0,
                            )
                            * f_poly
                            / (Q)
                            * P_press_species[self.monomers[i]]
                        )
                        * self.grid.dV
                        / self.grid.V
                    )
                    hold_dens_part += (
                        cp.sum(
                            cp.sum(
                                (Q_c.T * polymer.h_struct).T[
                                    polymer.struct == self.monomers[i]
                                ],
                                axis=0,
                            )
                            * f_poly
                            / (Q)
                            * P_press_species[self.monomers[i]]
                        )
                        * self.grid.dV
                        / self.grid.V
                    )

        # compute solvent densities
        for solvent in self.solvent_dict:
            idx = self.monomers.index(solvent)
            exp_w_S = cp.exp(-P_species[self.monomers[idx]] / self.N)
            Q_S = cp.sum(exp_w_S) / (self.grid.k2.size)
            self.phi_all[idx] += exp_w_S * self.solvent_dict[solvent] / (self.N * Q_S)
            self.Q_dict[solvent] = cp.copy(Q_S)
            if for_pressure:
                self.dQ_dV_dict[solvent] = (
                    cp.sum(
                        (exp_w_S * self.solvent_dict[solvent] / (self.N * Q_S))
                        * P_press_species[solvent]
                    )
                    * self.grid.dV
                    / self.grid.V
                )

        # check if we are using salts
        if not self.use_salts:
            return

        phi_salt_shape = list(self.w_all.shape)
        phi_salt_shape[0] = 2
        self.phi_salt = cp.zeros(phi_salt_shape, dtype=complex)

        self.get_salt_concs()

        for i in range(len(self.salts)):
            if self.c_s == 0:
                break

            salt_conc = self.salt_concs[self.salts[i]]
            w_salt = self.salts[i].charge * self.gaussian_smear(
                self.psi, self.psi_smear
            )
            exp_w_salt = cp.exp(-w_salt / self.N)
            Q_salt = cp.sum(exp_w_salt) / (self.grid.k2.size)
            self.phi_salt[i] = (exp_w_salt * salt_conc / (Q_salt)) / self.N
            self.Q_dict[self.salts[i]] = Q_salt
            if for_pressure:
                w_press_salt = self.salts[i].charge * self.convolve(self.psi, gauss_16)
                self.dQ_dV_dict[self.salts[i]] = (
                    cp.sum((exp_w_salt * salt_conc / (self.N * Q_salt)) * w_press_salt)
                    * self.grid.dV
                    / self.grid.V
                )

        if self.has_nanps:
            for nanp in self.nanps:
                idx = self.monomers.index(nanp.type)
                dens = nanp.density * 1
                self.phi_all[idx] += dens
        return

    def get_salt_concs(self) -> None:
        """
        Computes and assigns the non-FH salt concentrations needed to maintain charge
        balance.

        Updates the following attributes:

        - `self.salt_concs`

        Raises
        ------
        ValueError
            Raised if the amount of total salt is not enough to correct the charge
            imbalance.
        """

        self.salt_concs = {}

        net_saltless_charge = self.get_net_saltless_charge()

        salt_charges = [salt.charge for salt in self.salts]
        if (
            net_saltless_charge < self.c_s * self.grid.V * min(salt_charges) / self.N
            or net_saltless_charge > self.c_s * self.grid.V * max(salt_charges) / self.N
        ):
            print(
                "Salt needed is ",
                abs(net_saltless_charge) / (self.grid.k2.size),
            )
            raise ValueError("Inadequate salt to correct charge imbalance")

        for salt in self.salts:
            self.salt_concs[salt] = (
                self.c_s - net_saltless_charge * self.N / (salt.charge * self.grid.V)
            ) / 2

    def reindex_Q_c(self, Q_c: cp.ndarray) -> cp.ndarray:
        """
        Function to reindex Q_c to correctly handle edges of polymers.

        Because the integration takes place over the polymer beads but the
        values are recorded at the joints, we need to reindex the joints back
        to the beads. Currently only implements a trapezoid-rule like integration.

        Parameters
        ----------

        Q_c
            $Q(s)$ with a shape that is associated with the joints.

        Returns
        -------

        new_Q_c
            $Q(s)$ one segment shorter than the input, with summation consistent with a
            continuous Gaussian chain.

        """

        shape = list(Q_c.shape)
        shape[0] -= 1
        new_Q_c = cp.zeros(shape, dtype=complex)
        new_Q_c += Q_c[1:] / 2 + Q_c[:-1] / 2

        return new_Q_c
