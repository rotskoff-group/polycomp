"""
!!! warning "Experimental Module"

    This module implements a Gibbs ensemble method for simulating phase
    coexistence. It is considered semi-experimental. It successfully reproduces the
    correct results for implicit solvent coacervation with 1 FH parameter, but all other
    features are of unknown accuracy. Specifically it is known to incorrectly handle
    pressure for semi-implicit solvents, and the advanced modules are not fully tested.

    Use with caution and please validate your results carefully. This code is
    provided as a starting point for advanced users wishing to implement Gibbs Ensemble
    themselves.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Tuple

import cupy as cp

if TYPE_CHECKING:
    from .complex_langevin_ETD import CL_RK2
    from .ft_system import PolymerSystem
    from .grid import Grid


class GibbsEnsemble(object):
    """Manages a two-box Gibbs Ensemble simulation for phase coexistence.

    This class holds two `PolymerSystem` instances, allowing them to
    exchange volume and particles to find phase equilibrium. It handles the
    dynamics of the Gibbs ensemble moves based on differences in pressure and
    chemical potentials between the two boxes.

    Parameters
    ----------
    ps_1
        A fully initialized `PolymerSystem` for the first box. Its parameters
        (e.g., FH matrix) are used as a template for the second box.
    integrator_1
        A fully initialized integrator corresponding to `ps_1`.
    del_t
        The fictitious time step for mass transfer moves
        between the boxes.
    V_t
        The fictitious time step for volume exchange moves
        between the boxes.
    spec_dict_2
        Initial species concentrations for the second box. If `None`, the
        concentrations from `ps_1` are copied. Default is `None`.
    grid_2
        An initial `Grid` object for the second box. If `None`, the grid from
        `ps_1` is deep-copied. Default is `None`.
    salt_conc_2
        Initial salt concentration for the second box. If `None`, the value
        from `ps_1` is used. Default is `None`.
    integrator_2
        An integrator for the second box. If `None`, the integrator from `ps_1`
        is deep-copied. Default is `None`.

    Attributes
    ----------
    part_1 : PolymerSystem
        The `PolymerSystem` object for the first simulation box.
    part_2 : PolymerSystem
        The `PolymerSystem` object for the second simulation box.
    int_1 : CL_RK2
        The integrator for the first box.
    int_2 : CL_RK2
        The integrator for the second box.
    gibbs_t : float
        The fictitious time step for mass transfer moves.
    V_t : float
        The fictitious time step for volume exchange moves.
    species : tuple
        An ordered tuple of all species objects (`Monomer`, `Polymer`, `Salt`)
        managed by the ensemble.
    salted : bool
        `True` if the system contains salt species.
    mass_1 : cp.ndarray
        A vector containing the total mass (volume fraction * box volume) of
        each species in the first box. Indexed according to `self.species`.
    mass_2 : cp.ndarray
        A vector containing the total mass of each species in the second box.
    total_mass : cp.ndarray
        The conserved total mass of each species across both boxes
    total_V : float
        The conserved total volume of the two boxes (`part_1.grid.V` + `part_2.grid.V`).
    C_1 : cp.ndarray
        A vector of the concentrations of each species in the first box.
    C_2 : cp.ndarray
        A vector of the concentrations of each species in the second box.
    charge_vector : cp.ndarray
        A vector containing the charge of each species, used for ensuring
        charge-neutral mass transfer moves.
    sampled_flag : bool
        An internal flag set to `True` after `sample_pot` has been run,
        indicating that new chemical potential and pressure data is available.
    d_mu : cp.ndarray
        The calculated difference in chemical potentials (`mu_2 - mu_1`) for
        each species after sampling. Used to drive the `gibbs_step`.
    d_pi : float
        The calculated difference in pressure (`pi_2 - pi_1`) after sampling.
        Used to drive the `gibbs_step`.
    sampled_pot_1 : cp.ndarray
        The time-averaged chemical potential for each species in the first box,
        calculated by `sample_pot`.
    sampled_pot_2 : cp.ndarray
        The time-averaged chemical potential for each species in the second box.
    sampled_pressure_1 : float
        The time-averaged pressure in the first box, calculated by `sample_pot`.
    sampled_pressure_2 : float
        The time-averaged pressure in the second box.
    bound_list : list, optional
        (Experimental) A list of binding matrices used by `bind_species` to
        constrain mass transfer moves.
    """

    def __init__(
        self,
        ps_1: PolymerSystem,
        integrator_1: CL_RK2,
        del_t: float,
        V_t: float,
        spec_dict_2: dict | None = None,
        grid_2: Grid | None = None,
        salt_conc_2: float | None = None,
        integrator_2: CL_RK2 | None = None,
    ) -> None:

        self.gibbs_t = del_t
        self.V_t = V_t
        # take one system and copy it to retain same FH params,
        # polymers, monomers, etc
        self.part_1 = ps_1
        self.part_1.get_densities()
        self.part_2 = copy.copy(ps_1)

        # Need explicit copies for variable parts
        self.part_2.solvent_dict = copy.copy(self.part_1.solvent_dict)
        self.part_2.poly_dict = copy.copy(self.part_1.poly_dict)
        self.part_2.w_all = copy.copy(self.part_1.w_all)
        self.part_2.psi = copy.copy(self.part_1.psi)
        self.part_2.phi_all = copy.copy(self.part_1.phi_all)
        self.part_2.Q_dict = copy.copy(self.part_1.Q_dict)

        # default is to start with same grid size for both
        if grid_2 is None:
            self.part_2.grid = copy.deepcopy(self.part_1.grid)
        # allows for different starting grids to be used
        else:
            self.part_2.grid = grid_2

        # takes arguments for establishing integrator and builds internally
        self.int_1 = integrator_1
        if integrator_2 is None:
            self.int_2 = copy.deepcopy(integrator_1)
        else:
            self.int_2 = integrator_2

        self.int_2.ps = self.part_2

        # set the concentration for spec_dict_2
        if spec_dict_2 is not None:
            for dic in (self.part_2.poly_dict, self.part_2.solvent_dict):
                for spec in spec_dict_2:
                    if spec in dic:
                        dic[spec] = spec_dict_2[spec]

        if hasattr(self.part_1, "salts"):
            self.salted = True
        else:
            self.salted = False

        # set the salt concentration for part_2
        if salt_conc_2 is not None and self.salted:
            self.part_2.c_s = salt_conc_2

        self.part_1.get_densities()
        species = []
        for i in self.part_1.Q_dict:
            species.append(i)
        self.species = tuple(species)
        self.get_current_state()

        self.sampled_flag = False

        # get the charge vector
        self.get_charge_vector()

    def __repr__(self):
        return NotImplemented

    def get_current_state(self) -> None:
        """Recalculates and updates the mass and concentration vectors for both boxes.

        This method should be called if the underlying `PolymerSystem` objects
        (`part_1` or `part_2`) have been modified externally. It reads the current
        volumes and species concentrations to ensure the ensemble's internal
        state is synchronized. It is called automatically by `gibbs_step`.

        Notes
        -----
        This method modifies the `GibbsEnsemble` object in-place.

        This method updates the following attributes:
        - `self.mass_1`, `self.mass_2` (Total mass of each species in each box)
        - `self.total_mass` (Conserved total mass of each species)
        - `self.total_V` (Conserved total volume)
        - `self.C_1`, `self.C_2` (Concentration of each species in each box)
        """

        self.mass_1 = cp.zeros(len(self.species))
        self.mass_2 = cp.zeros(len(self.species))
        ps = self.part_1
        # TODO: remove later
        # ps.get_densities()
        for i in ps.poly_dict:
            self.mass_1[self.species.index(i)] += ps.poly_dict[i] * ps.grid.V
        for i in ps.solvent_dict:
            self.mass_1[self.species.index(i)] += ps.solvent_dict[i] * ps.grid.V
        if self.salted:
            for salt in ps.salts:
                self.mass_1[self.species.index(salt)] += ps.salt_concs[salt] * ps.grid.V

        ps = self.part_2
        # TODO: can be removed later
        # ps.get_densities()

        for i in ps.poly_dict:
            self.mass_2[self.species.index(i)] += ps.poly_dict[i] * ps.grid.V
        for i in ps.solvent_dict:
            self.mass_2[self.species.index(i)] += ps.solvent_dict[i] * ps.grid.V
        if self.salted:
            ps.get_salt_concs()
            for salt in ps.salts:
                self.mass_2[self.species.index(salt)] += ps.salt_concs[salt] * ps.grid.V

        self.total_mass = self.mass_1 + self.mass_2
        self.total_V = self.part_1.grid.V + self.part_2.grid.V
        self.C_1 = self.mass_1 / self.part_1.grid.V
        self.C_2 = self.mass_2 / self.part_2.grid.V

    def get_chemical_potential(self) -> None:
        """Calculates pressure and chemical potential differences from sampled data.

        This method processes the time-averaged values populated by `sample_pot` to
        calculate the thermodynamic driving forces ($\\Delta\\mu$ and $\\Delta\\Pi$) for
        the Gibbs ensemble moves.

        This method updates the following attributes:

        - `self.d_mu`
        - `self.d_pi`
        - `self.sampled_flag` (Resets to `False`)

        Notes
        -----
        This method modifies the `GibbsEnsemble` object in-place. It is called
        automatically by `gibbs_step` and requires `sample_pot` to have been
        run first.

        Raises
        ------
        RuntimeError
            If `sample_pot` has not been called and no sampled data is available.

        """

        # TODO: use average rather than last point to reduce noise
        self.d_mu = cp.zeros(len(self.species))
        if self.sampled_flag:
            self.d_mu = self.sampled_pot_2 - self.sampled_pot_1
            self.d_pi = self.sampled_pressure_2 - self.sampled_pressure_1
            self.sampled_flag = False
        #            print("using sampled data")
        else:
            return NotImplemented

        self.d_mu = self.d_mu.real
        self.d_pi = self.d_pi.real
        #        print("d_mu")
        #        print(self.d_mu)
        #        print("raw mu")
        #        print(self.sampled_pot_1)
        #        print(self.sampled_pot_2)
        #        print("d_pi")
        #        print(self.d_pi)
        #        print("raw pi")
        #        print(self.sampled_pressure_1)
        #        print(self.sampled_pressure_2)

        if hasattr(self, "bound_list"):
            for b in self.bound_list:
                bound_mu = b * self.d_mu / cp.sum(b)
                self.d_mu[b != 0] = bound_mu[b != 0]

    #            print("Bound d_mu")
    #            print(self.sampled_pot_1)
    #            print(self.sampled_pot_2)
    #            print(self.d_mu)

    def gibbs_step(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Performs one Gibbs ensemble move to update volumes and particle numbers.

        This is the core method for evolving the Gibbs ensemble. It calls
        `get_current_state` and `get_chemical_potential` to get the thermodynamic
        driving forces, then proposes a move to exchange volume and mass between
        the two boxes.

        The method includes safety checks to prevent negative masses or volumes
        and can use `neutral_charge_step` for charged systems. It modifies the
        internal `part_1` and `part_2` `PolymerSystem` objects in-place.

        Returns
        -------
        new_volumes : cp.ndarray
            A CuPy array of shape (2,) containing the new volumes of box 1 and box 2.
        new_concentrations : cp.ndarray
            A CuPy array of shape (2, n_species) containing the new concentrations
            for each species in each box.
        """

        # Update all the current mass and mu before updating
        self.get_current_state()
        self.get_chemical_potential()
        # self.get_change()

        safety_minimum = 0.5
        if cp.all(self.charge_vector == 0):
            new_m_1 = self.mass_1 + self.gibbs_t * self.d_mu
            new_m_2 = self.total_mass - new_m_1

            # implement a routine to check if any of the new masses are negative and
            # replace them with half their original value if they are

            if cp.any(new_m_1 < self.mass_1 * safety_minimum):
                where = new_m_1 < self.mass_1 * safety_minimum
                new_m_1[where] = self.mass_1[where] * safety_minimum
                new_m_2 = self.total_mass - new_m_1
                print("WARNING: Mass safety triggered")
            if cp.any(new_m_2 < self.mass_2 * safety_minimum):
                where = new_m_2 < self.mass_2 * safety_minimum
                new_m_2[where] = self.mass_2[where] * safety_minimum
                new_m_1 = self.total_mass - new_m_2
                print("WARNING: Mass safety triggered")
        else:
            new_m_1, new_m_2 = self.neutral_charge_step(self.d_mu)

        new_V_1 = self.part_1.grid.V - self.V_t * self.d_pi * self.total_V
        new_V_2 = self.total_V - new_V_1
        if new_V_1 < self.part_1.grid.V * safety_minimum:
            new_V_1 = self.part_1.grid.V * safety_minimum
            new_V_2 = self.total_V - new_V_1
            print("WARNING: Volume safety triggered")
        if new_V_2 < self.part_2.grid.V * safety_minimum:
            new_V_2 = self.part_2.grid.V * safety_minimum
            new_V_1 = self.total_V - new_V_2
            print("WARNING: Volume safety triggered")

        print("Volumes:", "{:.3f}".format(new_V_1), "{:.3f}".format(new_V_2))

        # calculate concentrations
        #        new_C_1 = (new_m_1 / new_V_1)[:-2]
        #        new_C_2 = (new_m_2 / new_V_2)[:-2]

        new_C_1 = new_m_1 / new_V_1
        new_C_2 = new_m_2 / new_V_2
        print(self.d_mu)
        print("Mass changes")
        print(new_m_1 - self.mass_1)
        print(new_m_2 - self.mass_2)
        # exit()
        print("Concentrations:")
        print(new_C_1)
        print(new_C_2)
        print("Actual mass:")
        print("Total", new_C_1 * new_V_1 + new_C_2 * new_V_2)
        print("1:", new_C_1 * new_V_1)
        print("2:", new_C_2 * new_V_2)
        # update grid side lengths
        self.part_1.grid.update_l(
            self.part_1.grid.ndims * [new_V_1 ** (1 / self.part_1.grid.ndims)]
        )
        self.part_2.grid.update_l(
            self.part_1.grid.ndims * [new_V_2 ** (1 / self.part_2.grid.ndims)]
        )

        # update concentrations
        ps = self.part_1
        for i in ps.poly_dict:
            ps.poly_dict[i] = new_C_1[self.species.index(i)]
        for i in ps.solvent_dict:
            ps.solvent_dict[i] = new_C_1[self.species.index(i)]

        c_s = 0
        if ps.use_salts:
            for salt in ps.salts:
                c_s += new_C_1[self.species.index(salt)]
            ps.c_s = c_s

        ps = self.part_2
        for i in ps.poly_dict:
            ps.poly_dict[i] = new_C_2[self.species.index(i)]
        for i in ps.solvent_dict:
            ps.solvent_dict[i] = new_C_2[self.species.index(i)]

        c_s = 0
        if ps.use_salts:
            for salt in ps.salts:
                c_s += new_C_2[self.species.index(salt)]
            ps.c_s = c_s

        return cp.asarray([new_V_1, new_V_2]), cp.asarray([new_C_1, new_C_2])
        # TODO:Figure out how to handle salts

    def burn(self, steps: int) -> None:
        """Runs the simulation for a number of steps without sampling.

        This method is used for equilibration, allowing the system to relax
        before data collection begins. It evolves both boxes using their
        respective integrators. During this phase, pressure calculations are
        enabled to prepare the system for potential `gibbs_step` moves.

        Parameters
        ----------
        steps : int
            The number of integration steps to perform.
        """

        for i in range(steps):
            self.int_1.ETD(for_pressure=True)
            self.int_2.ETD(for_pressure=True)

    def sample_pot(self, steps: int, sample_freq: int = 1) -> None:
        """Runs the simulation and samples pressure and chemical potentials.

        This method evolves both boxes and averages the chemical potentials and
        pressures over a number of steps. The averaged results are stored
        internally and are required by the `gibbs_step` method to calculate
        the driving forces for mass and volume exchange.

        This method modifies the object state by setting the `sampled_flag` to `True`
        and populating the `sampled_pot_*` and `sampled_pressure_*` attributes.

        Parameters
        ----------
        steps : int
            The total number of integration steps to run.
        sample_freq : int, optional
            The frequency at which to sample data. For example, `sample_freq=10`
            means data is collected every 10 steps. Default is 1.
        """

        samples = 0
        tot_weighted_pot_1 = cp.zeros(len(self.total_mass), dtype=complex)
        tot_weighted_pot_2 = cp.zeros(len(self.total_mass), dtype=complex)
        self.sampled_pressure_1 = 0
        self.sampled_pressure_2 = 0

        for i in range(steps):
            self.int_1.ETD(for_pressure=True)
            self.int_2.ETD(for_pressure=True)

            if (i + 1) % sample_freq == 0:
                self.part_1.get_chemical_potential()
                self.part_2.get_chemical_potential()
                for i in self.part_1.chem_pot_dict:
                    tot_weighted_pot_1[
                        self.species.index(i)
                    ] += self.part_1.chem_pot_dict[i]
                for i in self.part_2.chem_pot_dict:
                    tot_weighted_pot_2[
                        self.species.index(i)
                    ] += self.part_2.chem_pot_dict[i]

                self.sampled_pressure_1 += self.part_1.get_pressure()
                #                self.part_2.get_densities(for_pressure=True)
                self.sampled_pressure_2 += self.part_2.get_pressure()
                samples += 1
        self.sampled_pot_1 = tot_weighted_pot_1 / samples
        self.sampled_pot_2 = tot_weighted_pot_2 / samples
        self.sampled_pressure_1 /= samples
        self.sampled_pressure_2 /= samples

        self.sampled_flag = True

    def get_charge_vector(self) -> None:
        """Computes and stores the charge vector for all species in the ensemble.

        This vector contains the total charge per molecule (or per solvent) for
        each species. It is used by `neutral_charge_step` to enforce charge
        neutrality during mass transfer moves.

        This method modifies the `GibbsEnsemble` object in-place. It is called
        automatically during initialization.

        This method updates the following attribute:
        - `self.charge_vector`
        """

        # get vector of all species

        self.charge_vector = cp.zeros(len(self.species))
        for species in self.species:
            if species in self.part_1.poly_dict:
                charge_struct = cp.array([s.charge for s in species.struct])
                self.charge_vector[self.species.index(species)] = (
                    cp.sum(charge_struct * species.h_struct) * species.total_length
                )
            if species in self.part_1.solvent_dict:
                self.charge_vector[self.species.index(species)] = species.charge
            if self.salted and species in self.part_1.salts:
                self.charge_vector[self.species.index(species)] = species.charge

    def neutral_charge_step(self, mu: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculates a charge-neutral mass transfer step for charged systems.

        This method projects the proposed mass transfer vector (driven by `mu`)
        onto the subspace of charge-neutral moves. It includes safety checks
        to prevent species masses from becoming negative.

        Parameters
        ----------
        mu : cp.ndarray
            The vector of chemical potential differences driving the mass transfer.

        Returns
        -------
        new_m_1 : cp.ndarray
            The proposed new mass vector for box 1, guaranteed to be charge-neutral.
        new_m_2 : cp.ndarray
            The proposed new mass vector for box 2.
        """
        if not hasattr(self, "charge_vector"):
            self.get_charge_vector()
        corr_term = cp.sum(mu * self.charge_vector) / cp.sum(self.charge_vector**2)
        corr_mu_old = mu - corr_term * self.charge_vector
        corr_mu = self.charge_correction(mu, 0)
        print("Comparing methods")
        print(corr_mu_old)
        print(corr_mu)

        proposed_move = self.gibbs_t * corr_mu

        # We need to check if the desired move is allowed (it will be prohibited if
        # it would bring the mass below the current mass times the safety minimum)

        safety_minimum = 0.5
        # implement a routine to check if any of the new masses are negative and
        # replace them with half their original value if they are
        safety_passed = False
        required_move = proposed_move
        forced_move_check = cp.zeros(len(self.total_mass), dtype=bool)

        # It is still a convenient feature to allow for proposed moves that exceed
        # the allowed volumes but are just scaled back
        counts = 0
        while safety_passed is False:
            new_m_1 = self.mass_1 + proposed_move
            new_m_2 = self.total_mass - new_m_1
            safety_passed = True
            if cp.any(new_m_1 < self.mass_1 * safety_minimum):
                # checks if a move is forbidden
                where = new_m_1 < self.mass_1 * safety_minimum
                forced_move_check[where] = True
                required_move[where] = (
                    self.mass_1[where] * safety_minimum - self.mass_1[where]
                )
                safety_passed = False
                print("WARNING: Mass safety triggered")
            if cp.any(new_m_2 < self.mass_2 * safety_minimum):
                where = new_m_2 < self.mass_2 * safety_minimum
                forced_move_check[where] = True
                required_move[where] = -(
                    self.mass_2[where] * safety_minimum - self.mass_2[where]
                )
                safety_passed = False
                print("WARNING: Mass safety triggered")
            if safety_passed is False:
                counts += 1
                # we need to resolve the problem of the forbidden moves while
                # constraining which species are allowed to move
                charge_change = cp.sum(
                    required_move[forced_move_check]
                    * self.charge_vector[forced_move_check]
                )
                # Find the total charge of the forced moves
                temp_charge_vector = self.charge_vector * cp.logical_not(
                    forced_move_check
                )

                # find a new correction that acts only on unforced species and
                # corrects the imbalance introduced by the forced moves
                corr_term = (
                    cp.sum(mu * self.gibbs_t * temp_charge_vector) + charge_change
                ) / cp.sum(temp_charge_vector**2)
                proposed_move = mu * self.gibbs_t - corr_term * temp_charge_vector
                proposed_move[forced_move_check] = required_move[forced_move_check]

            if cp.all(forced_move_check == True):  # noqa: E712
                raise ValueError(
                    "Move finder has failed, all moves are forced "
                    "and charge is unbalanced"
                )
            if counts > len(self.mass_1):
                raise ValueError("Move finder failed, infinite loop broken")

        if not cp.allclose(cp.sum(proposed_move * self.charge_vector), 0):
            raise ValueError("Proposed charge imbalanced move that was not corrected")

        # if the mass safeties are triggered then we must recalculate the best possible
        # move subject to the further constraint that the forbidden moves are returned
        # to their allowed level and fixed

        new_m_1 = self.mass_1 + proposed_move
        new_m_2 = self.total_mass - new_m_1
        # print(corr_mu)
        # print(cp.sum(corr_mu * self.charge_vector))

        # print(self.charge_vector)
        # print(self.mass_1)
        # print(self.mass_2)
        # we only need to do part1 because conservation of charge suggests the other
        # should automatically balance for free

        return new_m_1, new_m_2

    def charge_correction(self, mu: cp.ndarray, total_charge: float) -> cp.ndarray:
        """(Experimental) Alternative method for enforcing charge neutrality.

        This function attempts to correct the chemical potential differences to
        ensure that the resulting mass transfer move is charge-neutral.

        Parameters
        ----------
        mu : cp.ndarray
            The vector of chemical potential differences driving the mass transfer.
        total_charge : float
            The target total charge for the correction (likely always 0).

        Returns
        -------
        corr_mu : cp.ndarray
            The corrected vector of chemical potential differences.
        """

        corr_mu = cp.zeros_like(mu)
        for i in range(len(mu)):
            for j in range(i + 1, len(mu)):
                print(mu[i], mu[j])
                corr = (
                    mu[i] * self.charge_vector[i] + mu[j] * self.charge_vector[j]
                ) / (self.charge_vector[i] ** 2 + self.charge_vector[j] ** 2)
                print(corr)
                scale = cp.abs(self.charge_vector[i]) / cp.abs(self.charge_vector[j])
                if scale < 1:
                    scale = 1 / scale
                print(scale)
                corr_mu[i] += (mu[i] - corr * self.charge_vector[i]) * scale
                corr_mu[j] += (mu[j] - corr * self.charge_vector[j]) * scale
                print(
                    "Should be 0: ",
                    (mu[i] - corr * self.charge_vector[i]) * self.charge_vector[i]
                    + (mu[j] - corr * self.charge_vector[j]) * self.charge_vector[j],
                )
        return corr_mu

    def bind_species(self, species_1: Any, species_2: Any) -> None:
        """(Experimental) Constrains two species to move together during mass transfer.

        This method imposes a constraint that the ratio of `species_1` to
        `species_2` must remain constant in both boxes during Gibbs ensemble moves.
        This is useful for simulating molecules that should not be separated, such as
        polyions and their bound counterions, or for enforcing a fixed salt-to-polymer
        ratio.

        Multiple calls to this method will combine constraints if they share a
        common species. For example, binding A to B, and then B to C, will result
        in A, B, and C all being bound together.

        Parameters
        ----------
        species_1
            The first species object to bind (e.g., a `Monomer` or `Polymer`).
        species_2
            The second species object to bind.

        Raises
        ------
        ValueError
            If the two species do not have the same relative concentration in
            both boxes at the time of binding, which would violate the
            constraint from the start.
        """

        if (
            self.C_1[self.species.index(species_1)]
            / self.C_1[self.species.index(species_2)]
            != self.C_2[self.species.index(species_1)]
            / self.C_2[self.species.index(species_2)]
        ):
            print(
                self.C_1[self.species.index(species_1)]
                / self.C_1[self.species.index(species_2)]
            )
            print(
                self.C_2[self.species.index(species_1)]
                / self.C_2[self.species.index(species_2)]
            )
            raise ValueError(
                "Cannot bind species with different concentrations in each simulation"
            )
            return

        binding_matrix = cp.zeros(len(self.species), dtype=float)
        binding_matrix[self.species.index(species_1)] = 1
        binding_matrix[self.species.index(species_2)] = (
            self.C_1[self.species.index(species_1)]
            / self.C_1[self.species.index(species_2)]
        )
        binding_matrix /= cp.amin(binding_matrix[binding_matrix != 0])
        # declare an empty list to store the binding matrix if it doesn't already exist
        if not hasattr(self, "bound_list"):
            self.bound_list = []
            self.bound_list.append(binding_matrix)
            return
        matched = False
        for bind in self.bound_list:
            if cp.any(bind * binding_matrix != 0):
                combined_bind = self.combine_binds(bind, binding_matrix)
                self.bound_list.remove(bind)
                self.bound_list.append(combined_bind)
                matched = True
                break

        if not matched:
            self.bound_list.append(binding_matrix)
        else:
            self.clean_binds()

    def combine_binds(self, bind_1: cp.ndarray, bind_2: cp.ndarray) -> cp.ndarray:
        """(Internal) Merges two compatible binding matrices into a single constraint.

        This helper method is called by `bind_species` when a new binding
        constraint is added that overlaps with an existing one. It combines
        them into a single, unified constraint vector.

        Notes
        -----
        This is an experimental helper method and is not intended for direct use.

        Parameters
        ----------
        bind_1 : cp.ndarray
            The first binding constraint vector.
        bind_2 : cp.ndarray
            The second binding constraint vector.

        Returns
        -------
        new_bind : cp.ndarray
            The combined binding constraint vector.

        Raises
        ------
        ValueError
            If the two binding matrices are incompatible (e.g., have conflicting ratios)
            or do not share any common species.
        """

        if cp.all(bind_1 * bind_2 == 0):
            raise ValueError("Unrelated matrices cannot be combined")
            return
        ratio = (bind_1 / bind_2)[bind_1 * bind_2 != 0]
        if not cp.all(ratio == ratio[0]):
            raise ValueError("Tried to combine incompatible binding matrices")
            return

        new_bind = bind_1 / ratio[0] + bind_2 * (
            cp.ones_like(bind_2) - (bind_1 * bind_2 != 0)
        )
        new_bind /= cp.amin(new_bind[new_bind != 0])
        return new_bind

    def clean_binds(self) -> None:
        """(Internal) Consolidates the list of binding matrices.

        This method iterates through the list of binding constraints and merges
        any overlapping constraints until only a set of orthogonal constraints
        remains.

        .. warning::
            This feature is totally untested and may contain bugs.

        Notes
        -----
        This is an experimental helper method for `bind_species` and is not
        intended for direct use. It modifies the `self.bound_list` attribute
        in-place.
        """

        # WARNING: Totally untested

        clean = False
        while not clean:
            clean = True
            for bind in self.bound_list:
                for other_bind in self.bound_list:
                    if bind is not other_bind:
                        if cp.all(bind * other_bind != 0):
                            combined_bind = self.combine_binds(bind, other_bind)
                            self.bound_list.remove(bind)
                            self.bound_list.remove(other_bind)
                            self.bound_list.append(combined_bind)
                            clean = False
                            break
                break
