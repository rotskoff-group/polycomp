# General Usage: Building a Simulation

Setting up a field-theoretic simulation in `polycomp` requires defining the physical components of the system, their thermodynamic interactions, the spatial grid, and the dynamic parameters for the integrator. Below is a structural guide to the required parameters and how they are assembled into a working simulation script.

## 1. Defining Chemical Species
Every simulation begins by declaring the fundamental building blocks of the system. 

*   **Monomers:** Instantiate individual monomer types using the `Monomer` class. You must define a unique name and specify a charge if the system includes electrostatics (default is 0). You can also flag species as solvent or salt.
*   **Polymers:** Construct polymer architectures using the `Polymer` class. You must provide a unique name, the total relative chain length ($N$), and a block structure. The block structure is a list of tuples pairing previously defined `Monomer` objects with their fractional length along the chain (e.g., `[(A_mon, 0.5), (B_mon, 0.5)]`).

All physical parameters are non-dimensionalized relative to a reference polymer length $N$ and a statistical segment length $b$. Spatial variables (coordinates and box lengths) are scaled relative to the reference radius of gyration $R_g = b\sqrt{N/6}$. Contour lengths are defined as fractional lengths relative to $N$.

!!! important "Important note: correctly setting polymer length"
    Each polymer has a rated length $N$ and an absolute length which is $N$ times the sum of the fractional length along the chain. 
    The system will scale the polymer by length N, and will by default set $N$ to the greatest $N$ value among polymers.
    Using polymers with fractional lengths that do not sum to 1 can lead to unintentionally modeling parameter values that are different than what is expected because the system
    continues to use $N$ as the reference length. 
    If this warning does not make sense, ensure that the sum of length fraction parameters is always 1, and adjust polymer lengths by setting their N values, which is a safe
    general practice.

!!! important "Conserving parameters across simulations"
    When comparing between different simulations, it is important to use the same value of $N$, or carefully rescale all other parameters 
    if you want to model the same underlying physics. 
    Because many parameters are set in terms of $N$, such as the FH matrix, changing that parameter can lead to many changes in the system you are modeling, with little to no 
    apparent change in density profile (because all parameters were rescaled together). 
    If you need to compare between the same system at different polymer lengths in different systems, the best practice is to set $N$ manually to be the same value for both systems.
    

## 2. System Thermodynamics
Once the physical species exist, you must define how much of each is present and how they interact.

*   **Flory-Huggins Interactions:** Create a dictionary mapping pairwise interactions to their corresponding $\chi N$ values. 
The keys must be a `frozenset` containing the two `Monomer` objects being evaluated. 
Self-interaction terms will only have one entry. 
Every possible pair of volume-occupying monomers must have an entry.
*   **System Composition:** Define the overall density fractions of the system using a dictionary mapping the instantiated `Polymer` and solvent `Monomer` objects to their relative concentrations. 
It is standard practice to scale these values so that the total system density sums to 1.

!!! important "Setting total system concentration"
    Setting the total system concentration also couples to many of the effective behaviors of the system and changes the underlying system being compared to. 
    If making direct comparisons to MD simulations or other types of models, the value will often not sum to 1.
    If working purely with field theory models, setting total concentration to be 1 will make interpreting the parameters as simple as possible. 

## 3. Discretization and Geometry
The continuous fields must be projected onto a discrete grid for numerical integration.

*   **Grid Dimensions:** Define the physical size of the simulation box in units of the radius of gyration ($R_g$) using a tuple (e.g., `(90, 90)`).
*   **Grid Resolution:** Specify the number of lattice points along each axis using a tuple (e.g., `(256, 256)`).
This dictates the spatial resolution and determines the dimensionality of the system (1D, 2D, or 3D).
*   **Smearing Length:** Assign a Gaussian smearing constant ($a$) to regularize the density fields.
This should generally be comparable to the physical spacing between adjacent grid points.
While this parameter arises from avoiding UV divergences, it is of physical consequence in the simulation. 

## 4. Integration Parameters
After initializing the `PolymerSystem` object with the above configurations, you must configure the Complex Langevin integration dynamics.

*   **Relaxation Rates ($\lambda$):** Define the fictitious time step sizes for the chemical potential fields (`relax_rates`). 
These govern how quickly the fields evolve during gradient descent.
*   **Fictitious Temperatures ($\beta$):** Set the variance for the stochastic noise injected into the fields (`temps`). 
These can be used to determine which type of CL integration you conduct (in terms of which real and imaginary fields are noised). 
*   **Electrostatics:** If the system is charged, set the rescaled Bjerrum length ($E$). 
You must also specify an electrostatic relaxation rate and fictitious temperature. 
If modeling a neutral system, set $E$, `psi_rate`, and `psi_temp` to 0.

## 5. Execution
With the system and dynamic parameters defined, instantiate the integrator (e.g., `CL_RK2`). 
To evolve the simulation, place the integrator's evaluation method (`integrator.ETD()`) inside a loop. 
Data collection, density visualization, and trajectory saving can be executed at regular intervals within this loop to monitor the equilibration of the system.
To track convergence you can plot the free energy and observe how the density changes over time, both should reach a stable state as the system equilibrates. 
Examples of working systems are present in the examples file and are a good starting point for building your own simulations. 
