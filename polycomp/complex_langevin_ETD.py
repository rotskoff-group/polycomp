import cupy as cp
import cupyx.scipy.fft as cufft
import math
#import polycomp.neural_ansatz as ansatz

import sys 


import_flag=False
ansatz=None

class CL_RK2(object):
    """
    Class for storing all of the necessary functions to run the Complex
    Langevin integration using Exponential Time Differencing.

    Attributes:
        ps (PolymerSystem Object):
            The polymer system integration is supposed to occur on.
        relax_rates (cparray):
            cparray of floats for the relaxation rate of each species.
        relax_temps (cparray):
            cparray of floats for the relaxation temps of each diagonalized w.
        psi_relax_rate (float):
            Float for the relaxation rate of psi.
        psi_temp (float):
            Float for the relaxation temp of psi.
        E (float):
            Float for the E (rescaled Bjerrum length) of the system.
        c_k_w (cparray):
            cparray of complex128 for the linear approximation of the response of force from
            fields derived using the weak inhomogeneity expansion.
        c_k_w (cparray):
            cparray of complex128 for the linear approximation of the response of force from
            charge field derived using the weak inhomogeneity expansion.
    """

    def __init__(self, poly_sys, relax_rates, relax_temps, psi_relax_rate, psi_temp, E):
        """
        Initialize integrator.

        Parameters:
            poly_sys (PolymerSystem object):
                Polymer system that integration is supposed to occur on.
            relax_rates (cparray):
                cparray of floats for the relaxation rates of each diagonalized w.
            relax_temps (cparray):
                cparray of floats for the relaxation temps of each diagonalized w.
            psi_relax_rate (float):
                Float for the relaxation rate of psi.
            psi_temp (float):
                Float for the relaxation temp of psi.
            E (float):
                Float for the E (rescaled Bjerrum length) of the system.

        Raises:
            ValueError:
                Raised if the shape of the relax rates doesn't match the shape
                of poly_sys.
        """
 
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

    def ETD(self, for_pressure=False, neural_ansatz=False):
        """
        Integrate one step of time with ETD algorithm.

        Parameters:
            for_pressure (bool, optional):
                Boolean for whether or not the integration will be followed by pressure
                calculations. This adds about 25% to the runtime, so it should be set as rarely
                as possible. Default is False.
        """

        if for_pressure and neural_ansatz:
            raise ValueError("Can't calculate pressure from a neural ansatz")
        # Get the densities
        global import_flag, ansatz
        if neural_ansatz is True:
            if import_flag is False:
                import polycomp.neural_ansatz as ansatz
                import_flag=True
            ansatz.infer(self.ps)
        else:
            self.ps.get_densities(for_pressure=for_pressure)

        # generate the random noise array that is going to be used with
        # appropriate variance
        w_dens_noise = cp.zeros_like(self.ps.w_all, dtype=complex)
        psi_dens_noise = cp.zeros_like(self.ps.psi, dtype=complex)
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

        w_trans_noise = self.ps.map_norm_from_dens(w_dens_noise)

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
            -self.ps.smear_const * self.ps.grid.k2
        )

        if self.c_k_w is None:
            self.build_c_k(u0_eig, debye_k)

        for i in range(w_k.shape[0]):
            self.c_k_w[i].flat[0] = u0_eig[i]

        # Need to sum over degenerate modes and fourier transform density to
        # prepare for dynamics
        red_dens = self.ps.remove_degeneracy(self.ps.phi_all)
        red_dens = self.ps.gaussian_smear(red_dens, self.ps.smear_const)

        real_dens_k = self.fourier_along_axes(red_dens, 0)

        tot_charge = self.ps.get_total_charge()
        tot_charge = self.ps.gaussian_smear(tot_charge, self.ps.smear_const)
        tot_charge_k = cufft.fftn(tot_charge)

        # Generate the force trajectories
        F_k_w = (
            -self.ps.gamma**2
            * ((w_k.T / u0_eig) - self.ps.map_norm_from_dens(real_dens_k).T)
        ).T

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

    def fourier_along_axes(self, array, axis):
        """
        Fourier transform each grid separately along the first axis.

        Needed when each grid is stored as a stacked array. Uses cufft for
        Fourier transforms.

        Parameters:
            array (cparray):
                Array to be Fourier transformed over.
            axis (int):
                Axis to be treated separately.
        """

        f_array = cp.zeros_like(array, dtype=complex)

        # code to slice out everything correctly
        for i in range(array.shape[axis]):
            sl = [slice(None)] * array.ndim
            sl[axis] = i
            f_array[tuple(sl)] = cufft.fftn(array[tuple(sl)])
        return f_array

    def inverse_fourier_along_axes(self, array, axis):
        """
        Inverse Fourier transform each grid separately along the first axis.

        Needed when each grid is stored as a stacked array. Uses cufft for
        Inverse Fourier transforms.

        Parameters:
            array (cparray):
                Array to be inverse Fourier transformed over.
            axis (int):
                Axis to be treated separately.
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

    def debye(self, k2):
        """
        Debye function on a discrete grid.

        Parameters:
            k2 (cparray):
                cparray representing k^2 at each grid point in k-space.
        """

        debye = 2 / (k2**2) * (cp.exp(-k2) - 1 + k2)
        return debye

    def draw_gauss(self, variance):
        """
        Draw Gaussian distribution independently at each point in space.

        Parameters:
            variance (float):
                Variance of the Gaussian to be drawn.
        """

        return cp.random.normal(0, variance.real)

    def build_c_k(self, u0_eig, debye_k):
        """
        Build the c_k coefficients for the ETD integrator.

        Parameters:
            u0_eig (cparray):
                Eigenvalues of the u0 matrix.
            debye_k (cparray):
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
                # TODO: Honestly, this stuff is to complicated for me to follow and
                # should be done in greater detail with someone to check exactly
                # DANGER: Factor of 2 is from Villet 2014, but I didn't have it
                # in the derivation. Check this, all four instances of 2 *

                # This is the leading g_jj term
                self.c_k_w[ids[i]] += (
                    2
                    * fract
                    / (alphk**2)
                    * (alphk * segs[i] + cp.exp(-alphk * segs[i]) - 1)
                    * cp.exp(-alphk * self.ps.smear_const**2)
                )
                self.c_k_psi += (
                    2
                    * fract
                    / (alphk**2)
                    * (alphk * segs[i] + cp.exp(-alphk * segs[i]) - 1)
                    * polymer.block_structure[i][0].charge ** 2
                    * cp.exp(-alphk * self.ps.smear_const**2)
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
                        * cp.exp(-alphk * self.ps.smear_const**2)
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
                        * cp.exp(-alphk * self.ps.smear_const**2)
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
