# Mathematical Background

The polycomp package implements the CL-FTS method. This section provides a brief overview of the concepts behind the method and how they are implemented here. 
This package samples fluctuations of the field theory, rather than solving for the mean field solution, making it suitable for solving certain problems where 
the mean-field answer is incorrect, such as coacervation.

## 1. The Field-Theoretic Transformation

Field theory approaches start from standard, particle-based models of a type that could be implemented in traditional molecular dynamics software. 
These models need to be mapped from a discrete point-based system to a continuous, field based one. 
This mapping produces density profiles in space of the underlying chemical species.
The chain must be represented mathematically, which is done by using a continuous Gaussian chain model. 
The stretching energy for an isolated Gaussian chain is given by:

$$U_0[\boldsymbol{r}] = \frac{3k_B T}{2 N b^2} \int_0^N ds \left| \frac{d\boldsymbol{r}(s)}{ds} \right|^2$$

where $N$ is the reference polymer length, $b$ is the statistical segment length, and $s$ is the contour variable.

Pairwise interactions lead to complex forms of the energy that are difficult to compute. 
In the auxiliary field method, as implemented here, this is overcome by the Hubbard-Stratonovich transformation. 
This transformation replaces all of the density-density interactions with interactions purely between the new auxiliary fields themselves, or 
between the density and auxiliary fields, which simplifies the calculations. 

## 2. Regularization and Smeared Densities

Field theories are known to suffer from ultraviolet divergences if not regularized. 
`polycomp` coduncts this regularization via a Gaussian smearing kernel that distributes their density over some spatial extent:

$$\Gamma(\boldsymbol{r}) = (2\pi a^2)^{-d/2} \exp\left(-\frac{|\boldsymbol{r}|^2}{2a^2}\right)$$

where $a$ is the smearing length and $d$ is the dimensionality of the system. 
The local microscopic density $\bar{\rho}(\boldsymbol{r})$ is convoluted with this kernel to produce a smeared density:

$$\rho(\boldsymbol{r}) = \Gamma * \bar{\rho}(\boldsymbol{r}) = \int_{\Omega} \Gamma(|\boldsymbol{r} - \boldsymbol{r}'|) \bar{\rho}(\boldsymbol{r}') d\boldsymbol{r}'$$

This transform avoids the UV divergences and stabilizes the integration of the code. 

## 3. The Field-Theoretic Hamiltonian

The transformation yields a statistical field theory defined by an effective Hamiltonian, $H[\{\mu_i\}, \varphi]$. 
This functional depends on two types of auxiliary fields: 
chemical potential fields $\{\mu_i(\boldsymbol{r})\}$ for Flory-Huggins (FH) interactions, and electrostatic potential field $\varphi(\boldsymbol{r})$ 
for Coulombic interactions.

The Hubbard-Stratonovich transform can only act on terms that do not have cross interactions, so the original
FH interaction matrix $\boldsymbol{\chi}$ needs to be diagonalized. 
This produces the characteristic fields $\{\mu_i(\boldsymbol{r})\}$ which are in this diagonal basis. 
The field-theoretic Hamiltonian is formulated as:

$$H[\{\mu_i\}, \varphi] = \sum_{i=1}^{M} \frac{\gamma_i^2}{2B_i} \int_{\Omega} d\boldsymbol{r} \mu_i^2(\boldsymbol{r}) + \frac{1}{2E} \int_{\Omega} d\boldsymbol{r} 
|\nabla\varphi(\boldsymbol{r})|^2 - \sum_{j=1}^{P+S+2} n_j \log Q_j[\{\mu_i\}, \varphi] + \frac{V\boldsymbol{c}^T \boldsymbol{\chi} \boldsymbol{c}}{2}$$

Where:

* $M$ is the number of distinct monomer fields.
* $\gamma_i$ handles the sign of the FH eigenvalues (taking $1$ for $B_i > 0$ and $i$ for $B_i < 0$).
* $E$ is the rescaled Bjerrum length characterizing electrostatic strength.
* $n_j$ is the number of molecules of species $j$ (polymers, solvents, salts).
* $Q_j$ is the single-molecule partition function for species $j$.
* The final term is an analytic correction for the homogeneous field component.

## 4. The Modified Diffusion Equation (MDE)

To evaluate the Hamiltonian, we must compute the single-chain partition functions $Q_j$ and the local densities. 
Because we have decoupled the interactions, we can compute the single chain partition function to find the density. 
For a guassian chain subject to a chemical potential $\psi(\boldsymbol{r}, s)$, this can be solved via the Modified Diffusion Equation (MDE):

$$\frac{\partial q_j(\boldsymbol{r}, s)}{\partial s} = \nabla^2 q_j(\boldsymbol{r}, s) - \psi_j(\boldsymbol{r}, s) q_j(\boldsymbol{r}, s)$$

where $q_j(\boldsymbol{r}, s)$ is the statistical weight (propagator) of a chain segment of length $s$ ending at position $\boldsymbol{r}$. 
The local field $\psi_j$ felt by the monomer at contour $s$ is the effective chemical potential in the non-diagonalized basis:

$$\psi_j(\boldsymbol{r}) = \Gamma * \left( \boldsymbol{b}\boldsymbol{\mu}(\boldsymbol{r}) + Z_j \varphi(\boldsymbol{r}) \right)$$

The single-chain partition function is obtained by integrating the propagator over the system volume $V$:

$$Q_j = \frac{1}{V} \int_{\Omega} d\boldsymbol{r} q_j\left(\boldsymbol{r}, \frac{N_j}{N}\right)$$

By combining the forward propagator $q(\boldsymbol{r},s)$ and the reverse propagator $q^\dagger(\boldsymbol{r},s)$, we can compute the partition function
and corresponding density:

$$\rho(\boldsymbol{r}) = \frac{C_j}{Q_j} \int_0^{N_j/N} ds \, q_j(\boldsymbol{r}, s) q_j^\dagger(\boldsymbol{r}, s)$$

`polycomp` solves the MDE numerically using a pseudospectral method. 
The $\nabla^2$ operator is evaluated efficiently in Fourier space ($k$-space) using `cufft`, while the spatial field operator $\psi(\boldsymbol{r})$ is applied in real space. 
A fourth-order Richardson Extrapolation scheme applied to a Trotter decomposition is used to accurately integrate steps of $\Delta s$.

## 5. Complex Langevin Dynamics

Formally, getting the correct values of out this method requires sampling over all possible fields in $H[\{\mu_i\}, \varphi]$. 
However, we can find a tractable approximation by either finding the self-consistent mean field solution or by sampling around that solution. 
Because we are interested in observing coacervation, for which the mean-field solution is incorrect, we must turn to the sampling approach, implemented here as
complex Langevin (CL) sampling. 

All fields are sampled in their full, complex formuation and evolved in a fictitious time $t$ according to the Langevin equations:

$$\frac{\partial \mu(t, \boldsymbol{k})}{\partial t} = -\lambda_\mu \frac{\delta H}{\delta \mu(t, \boldsymbol{k})} + \gamma \odot \eta_\mu(t, \boldsymbol{k})$$

$$\frac{\partial \varphi(t, \boldsymbol{k})}{\partial t} = -\lambda_\varphi \frac{\delta H}{\delta \varphi(t, \boldsymbol{k})} + i \eta_\varphi(t, \boldsymbol{k})$$

Here, $\lambda$ represents the relaxation rates, and $\eta$ is Gaussian white noise with variance proportional to a fictitious temperature $\beta$. 
The gradients of the Hamiltonian with respect to the fields provide effective forces which push the system towards the lowest free energy state:

$$\frac{\delta H}{\delta \boldsymbol{\mu}} = \frac{\boldsymbol{\gamma}^2}{\boldsymbol{B}} \odot \boldsymbol{\mu}(\boldsymbol{r}) - \boldsymbol{b}^T \rho(\boldsymbol{r})$$

$$\frac{\delta H}{\delta \varphi} = -\frac{1}{E} \nabla^2 \varphi(\boldsymbol{r}) - \rho_C(\boldsymbol{r})$$

The noise terms, work to sample around that state. 

To address the numerical stiffness inherent in high-frequency Fourier modes, `polycomp` utilizes a first-order Exponential Time Differencing (ETD1) scheme. 
This explicitly integrates the linear response of the forces using an analytical approximation derived from the weak inhomogeneity expansion (Debye function), 
allowing the Langevin trajectories to be stable at larger integration time steps.


***

*Full mathematical descriptions of the underlying methods are available in: Pert, E. K., Hurst, P. J., Waymouth, R. M., & Rotskoff, G. M. (2025). 
"Coacervation drives morphological diversity of mRNA encapsulating nanoparticles", [The Journal of Chemical Physics, 162(7), 074902](https://doi.org/10.1063/5.0235799).*
