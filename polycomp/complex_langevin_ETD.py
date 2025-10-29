from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.fft as cufft

if TYPE_CHECKING:
    from ft_system import PolymerSystem


class CL_RK2(object):
    """
    Class used to update the state of a polymer field system using the complex langevin
    integrator with exponential time differencing. SCFT behavior can be recovered by
    setting noise to zero, though the integrators are less efficient than state of the
    art SCFT integrators. Can be set to compute pressure and density simultaneously for
    efficiency.

    Parameters
    ----------

    poly_sys
        Polymer system the integration scheme will be applied to.
    relax_rates
        Array of the relaxation rates for the chemical potential fields
        $\\{\\lambda_{\\omega}\\}$. Each entry corresponds to the same indexed field in
        the diagonal basis.
    relax_temps
        Array of the fictitious temperatures for the chemical potential fields
        $\\{\\lambda_{\\omega}\\}$. Each entry corresponds to the same indexed field in
        the diagonal basis.
    psi_relax_rate
        Relaxation rate of electric potential field $\\lambda_{\\varphi}$
    psi_temp
        Fictitious temperature for the electric potential field $\\beta_{\\varphi}$
    E
        Rescaled Bjerrum length of the system $E$.

    Attributes
    ----------

    ps : PolymerSystem
        Polymer system to be integrated.
    relax_rates : cupy.ndarray of floats
        Array of the relaxation rates for the chemical potential fields
        $\\{\\lambda_{\\omega}\\}$.
    relax_temps : cupy.ndarray of floats
        Array of the fictitious temperatures for the chemical potential fields
        $\\{\\lambda_{\\omega}\\}$. Each entry corresponds to the same indexed field in
        the diagonal basis.
    psi_relax_rate : float
        Relaxation rate of electric potential field $\\lambda_{\\varphi}$
    psi_temp : float
        Fictitious temperature for the electric potential field $\\beta_{\\varphi}$
    E : float
        Rescaled Bjerrum length of the system $E$.
    c_k_w : cupy.ndarray of complex
        Linear approximation of the response of force to change in $i$th chemical
        potential field $c_{i}(\\boldsymbol{k})$. Derived from weak inhomogeneity
        expansion with indexing to match the diagonal basis
    c_k_psi : cupy.ndarray of complex
        Linear approximation of the response of force to change in $i$th
        chemical potential field $c_{\\varphi}(\\boldsymbol{k})$. Derived from weak
        inhomogeneity expansion.

    Raises
    ------

    ValueError
        Raised if the shape of the relax rates doesn't match the shape
        of poly_sys.
    """

    def __init__(
        self,
        poly_sys: PolymerSystem,
        relax_rates: cp.ndarray,
        relax_temps: cp.ndarray,
        psi_relax_rate: float,
        psi_temp: float,
        E: float,
    ) -> None:

        super(CL_RK2, self).__init__()

        self.ps = poly_sys
        if len({self.ps.w_all.shape[0], len(relax_rates)}) != 1:
            raise ValueError("Wrong sized relax rate or temperature")
        self.relax_rates = relax_rates
        self.relax_temps = relax_temps
        self.psi_relax_rate = psi_relax_rate
        self.psi_temp = psi_temp
        self.E = E
        self.c_k_w = None

    def ETD(
        self, for_pressure: bool = False, for_inverse_problem: bool = False
    ) -> None:
        """
        Call to integrate the attached polymer system by one time step with the
        specified ETD integration parameters.

        Actual update steps are:

        $\\hat w (t + 1) = \\frac{1 - e^{-\\lambda c(\\boldsymbol{k})}}
        {c(\\boldsymbol{k})} \\hat F (w(t)) +
        (\\frac{1 -
                e^{-\\lambda c(\\boldsymbol{k})}}{2\\lambda c(\\boldsymbol{k})})^{1/2}
        \\hat \\eta (t)$

        where

        $\\hat F (\\boldsymbol{ \\mu }(t,\\boldsymbol{k})) =
        \\frac{\\boldsymbol{\\gamma}^2}{\\boldsymbol{B}} \\odot
        \\hat{\\boldsymbol{\\mu}}(t,\\boldsymbol{k}) -
        \\boldsymbol{b}^T \\hat{\\boldsymbol{\\rho}}(t,\\boldsymbol{k})$

        for the chemical potential fields and

        $\\hat F (\\varphi(t,\\boldsymbol{k})) =
        \\frac{1}{E}
        \\boldsymbol{k}^2 \\hat{\\varphi}(t,\\boldsymbol{k}) -
        \\boldsymbol{b}^T \\hat{\\rho}_C(t,\\boldsymbol{k})$

        for the electric field.

        The variance of the noise is
        $\\frac{2\\lambda_i\\beta_i}{\\Delta V}$ with the indexing for the same
        field.

        Parameters
        ----------

        for_pressure
            Flag for whether or not the integration will be followed by pressure
            calculations. This adds about 25% to the runtime, so it should not be set
            unless pressure is desired.
        for_inverse_problem
            Boolean for whether the method doesn't calculate new densities to allow
            for solving the inverse field problem - SCF solution for finding fields
            corresponding to a given density.
        """

        # Get the densities
        if not for_inverse_problem:
            self.ps.get_densities(for_pressure=for_pressure)

        # This is a temporary solution to handle the c_k term, which doesn't really
        # matter, but we'll just use the first entry of the smearing matrix as the
        # generic smear term for c_k
        self.smear_const = self.ps.smear_arr[0, 0]

        # generate the random noise array that is going to be used with
        # appropriate variance
        w_dens_noise = cp.zeros_like(self.ps.w_all, dtype=complex)
        for i in range(w_dens_noise.shape[0]):
            w_dens_noise[i] = (
                self.draw_gauss(
                    cp.full(
                        w_dens_noise[i].shape,
                        cp.sqrt(
                            2
                            * self.relax_rates[i]
                            * self.relax_temps[i]
                            / self.ps.grid.dV
                        ),
                    )
                )
                * self.ps.gamma[i]
            )

        psi_noise = (
            self.draw_gauss(
                cp.full(
                    w_dens_noise[i].shape,
                    cp.sqrt(2 * self.psi_relax_rate * self.psi_temp / self.ps.grid.dV),
                )
            )
            * 1j
        )

        #        w_trans_noise = self.ps.map_norm_from_dens(w_dens_noise)
        w_trans_noise = w_dens_noise
        # This is the correct operation, the noise comes in with the right symmetry and
        # transforming it further does weird things

        d_w = self.relax_rates
        d_psi = self.psi_relax_rate

        R_k_w = self.fourier_along_axes(w_trans_noise, 0)
        R_k_psi = cufft.fftn(psi_noise)

        w_k = self.fourier_along_axes(self.ps.normal_w, 0)
        psi_k = cufft.fftn(self.ps.psi)

        # Start preparing the linear approximation term used in ETD1
        u0_eig = self.ps.normal_evalues

        # Debye function is the linear approximation using weak inhomogeneity
        # expansion
        debye_k = self.debye(self.ps.grid.k2) * cp.exp(
            -self.smear_const * self.ps.grid.k2
        )

        if self.c_k_w is None:
            self.build_c_k(u0_eig, debye_k)

        for i in range(w_k.shape[0]):
            self.c_k_w[i].flat[0] = u0_eig[i]

        # Need to sum over degenerate modes and fourier transform density to
        # prepare for dynamics
        red_dens = self.ps.remove_degeneracy(self.ps.phi_all)
        red_dens = self.ps.map_norm_from_dens_smeared(red_dens)
        #        red_dens = self.ps.gaussian_smear(red_dens, self.ps.smear_arr[0,0])

        real_dens_norm_k = self.fourier_along_axes(red_dens, 0)

        tot_charge = self.ps.get_total_charge()
        tot_charge = self.ps.gaussian_smear(tot_charge, self.ps.psi_smear)
        tot_charge_k = cufft.fftn(tot_charge)

        # Generate the force trajectories
        F_k_w = (-self.ps.gamma**2 * ((w_k.T / u0_eig) - real_dens_norm_k.T)).T

        F_k_psi = psi_k * self.ps.grid.k2 / self.E - tot_charge_k

        # Run ETD step
        new_w_k = cp.zeros_like(w_k, dtype=complex)
        for i in range(w_k.shape[0]):
            new_w_k[i] = (
                w_k[i]
                - ((1 - cp.exp(-d_w[i] * self.c_k_w[i])) / self.c_k_w[i]) * F_k_w[i]
                + (
                    (1 - cp.exp(-2 * d_w[i] * self.c_k_w[i]))
                    / (2 * d_w[i] * self.c_k_w[i])
                )
                ** (1 / 2)
                * R_k_w[i]
            )

        # First element will be undefined, just set it to be unchanged
        for i in range(new_w_k.shape[0]):
            new_w_k[i].flat[0] = w_k[i].flat[0]
        #            new_w_k[i].flat[0] = w_k[i].flat[0] - F_k_w[i].flat[0] * d_w[i]
        #        print(new_w_k[:,0,0])

        new_psi_k = (
            psi_k
            - ((1 - cp.exp(-d_psi * self.c_k_psi)) / self.c_k_psi) * F_k_psi
            + ((1 - cp.exp(-2 * d_psi * self.c_k_psi)) / (2 * d_psi * self.c_k_psi))
            ** (1 / 2)
            * R_k_psi
        )

        # First element will be undefined, just set it to be unchanged
        new_psi_k.flat[0] = 0

        # Map back to real space
        new_w = self.inverse_fourier_along_axes(new_w_k, 0)
        new_psi = cufft.ifftn(new_psi_k, new_psi_k.shape)

        self.ps.normal_w = new_w
        self.ps.update_density_from_normal()

        # Set psi to zero if E is zero
        if self.E == 0:
            self.ps.psi = cp.zeros_like(self.ps.psi, dtype=complex)
            return

        self.ps.psi = new_psi

    def fourier_along_axes(self, array: cp.ndarray, axis: int) -> cp.ndarray:
        """
        Fourier transform each grid separately along the first axis.

        Needed when each grid is stored as a stacked array. Uses cufft for
        Fourier transforms.

        Parameters
        ----------

        array
            Stacked array to be Fourier transformed.
        axis
            Axis to be treated separately.

        Returns
        -------

        f_array : cp.ndarray
            Stacked array that is the fourier transform of the input array
        """

        f_array = cp.zeros_like(array, dtype=complex)

        # code to slice out everything correctly
        for i in range(array.shape[axis]):
            sl = [slice(None)] * array.ndim
            sl[axis] = i
            f_array[tuple(sl)] = cufft.fftn(array[tuple(sl)])
        return f_array

    def inverse_fourier_along_axes(self, array: cp.ndarray, axis: int) -> cp.ndarray:
        """
        Inverse Fourier transform each grid separately along the first axis.

        Needed when each grid is stored as a stacked array. Uses cufft for
        inverse Fourier transforms.

        Parameters
        ----------

        array
            Stacked arrays to be inverse Fourier transformed.
        axis
            Axis to be treated separately.

        Returns
        -------

        inf_array : cp.ndarray
            Stacked array that is the inverse Fourier transform of the input array
        """

        inf_array = cp.zeros_like(array, dtype=complex)

        # code to slice out everything correctly
        for i in range(array.shape[axis]):
            sl = [slice(None)] * array.ndim
            sl[axis] = i
            inf_array[tuple(sl)] = cufft.ifftn(
                array[tuple(sl)], s=array[tuple(sl)].shape
            )
        return inf_array

    def debye(self, k2: cp.ndarray) -> cp.ndarray:
        """
        Generates the Debye function on a discrete grid according to

        $\\hat{g}_{D}(k^2) = \\frac{2}{k^4} \\left( e^{-k^2} + k^2 - 1 \\right)$.

        Parameters
        ----------

        k2
            Array for $k^2$ at every point in the corresponding k space array.

        Returns
        -------

        debye : cp.ndarray
            The array for the Debye function at all points in space
        """

        debye = 2 / (k2**2) * (cp.exp(-k2) - 1 + k2)
        return debye

    def draw_gauss(self, variance: float) -> cp.ndarray:
        """
        Draws Gaussian distribution independently at each point in space.

        Parameters
        ----------

        variance
            Variance of the Gaussian to be drawn.

        Returns
        -------

        gaussian_array : cp.ndarray
            Array of random values
        """
        return cp.random.normal(0, variance.real)

    def build_c_k(self, u0_eig: cp.ndarray, debye_k: cp.ndarray) -> None:
        """
        Generates $c(\\boldsymbol{k})$ for all the polymer structures in the
        system given the FH matrix. Sets them as the matching appropriate internal
        variables for all fields.

        Parameters
        ----------

        u0_eig
            Eigenvalues of the u0 matrix.
        debye_k
            Fourier-transformed Debye function.

        """

        self.c_k_w = cp.zeros_like(self.ps.normal_w)
        self.c_k_psi = self.ps.grid.k2 / self.E

        # For every polymer we need to build a corresponding c_k, the procedure
        # is fully desrcibed in the extended derivation
        for polymer in self.ps.polymers:
            segs = cp.array([x[1] for x in polymer.block_structure], dtype=float)
            segs *= polymer.total_length / cp.sum(segs) / self.ps.N
            ids = cp.array(
                [self.ps.monomers.index(x[0]) for x in polymer.block_structure]
            )
            fract = self.ps.poly_dict[polymer]
            alphk = self.ps.grid.k2
            for i in range(len(polymer.block_structure)):
                # Reference for full derivation is present in Emmit Pert's PhD thesis

                # This is the leading g_jj term
                self.c_k_w[ids[i]] += (
                    2
                    * fract
                    / (alphk**2)
                    * (alphk * segs[i] + cp.exp(-alphk * segs[i]) - 1)
                    * cp.exp(-alphk * self.smear_const**2)
                )
                self.c_k_psi += (
                    2
                    * fract
                    / (alphk**2)
                    * (alphk * segs[i] + cp.exp(-alphk * segs[i]) - 1)
                    * polymer.block_structure[i][0].charge ** 2
                    * cp.exp(-alphk * self.smear_const**2)
                )

                for j in range(len(segs)):
                    if j == i:
                        continue
                    if abs(i - j) == 1:
                        gap = 0
                    elif i < j:
                        gap = cp.sum(segs[i + 1 : j])
                    else:
                        gap = cp.sum(segs[i + 1 : j])
                    # This is the g_ij term
                    self.c_k_w[ids[i]] += (
                        2
                        * fract
                        * cp.exp(-alphk * gap)
                        / (alphk**2)
                        * (1 - cp.exp(-alphk * segs[i]))
                        * (1 - cp.exp(-alphk * segs[j]))
                        * cp.exp(-alphk * self.smear_const**2)
                    )
                    self.c_k_psi += (
                        2
                        * fract
                        * cp.exp(-alphk * gap)
                        / (alphk**2)
                        * (1 - cp.exp(-alphk * segs[i]))
                        * (1 - cp.exp(-alphk * segs[j]))
                        * polymer.block_structure[i][0].charge
                        * polymer.block_structure[j][0].charge
                        * cp.exp(-alphk * self.ps.psi_smear**2)
                    )

        # c_k_w is the linear term used to set the scale of the dynamics
        self.c_k_w = self.ps.map_norm_from_dens(self.c_k_w)
        self.c_k_w = (self.c_k_w.T + 1 / u0_eig).T

        # Overwrite the zero mode term of each array
        sl = [slice(1)] * self.c_k_w.ndim
        sl[0] = slice(None)
        insert_u0 = cp.expand_dims(1 / u0_eig, axis=tuple(range(1, self.c_k_w.ndim)))
        self.c_k_w[tuple(sl)] = insert_u0

        # In this case psi is going to get wiped anyways but we don't want division by
        # zero
        if self.E == 0:
            return
        self.c_k_psi.flat[0] = 1 / self.E

        return
