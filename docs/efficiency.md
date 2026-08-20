# Efficient usage of `polycomp`

Polycomp should be useable for a wide array of useful simulation regimes. This section includes tips and expectations for efficient use of the code. 

!!! important "Precision"
    `polycomp` runs exclusively in complex double precision. The code is not consistently stable in lower precision due to the tight convergence required for computing
    the partition function. 

## General principle: Most cost is FFT

The most expensive operation is the FFT block, which has $O(n\log(n))$ scaling in the number of grid points, and which will usually comprise $>85\%$ of 
computation time in large simulations. As such, the strongest proxy for computational expense will be the number of grid points used, which for 3D simulations practically scales 
as roughly the cube of each side length. This is why most development is preferred in 2D, where fine grids can still be integrated quickly. 

## General principle: Most other costs are linear

For most other parameters aside from grid size, the cost should be roughly linear in each parameter. Example parameters for which the cost is roughly linear include:

* Number of time steps
* Number of total polymer integration points
    * Every distinct polymer has to be integrated separately, so adding different polymer types will increase runtime. 
* Number of distinct species types
    * The code will automatically combine degenerate species types in its representation, so there is no cost to declaring multiple species with identical FH interaction
profiles.


These costs are mostly unavoidable, with the exception of optimizing the time stepping rate to be as large as possible for stability and modifying the coarseness of the polymer
backbone integration. 

As this is GPU code, standard best practices about GPU utilization apply. Using grid sizes in powers of 2 is highly recommended. 

## Total cost and the memory problem

In these simulations, GPU VRAM limits are typically exhausted long before execution time becomes a prohibitive bottleneck. 
Memory requirements scale linearly with the number of contour integration points, non-degenerate species, and total grid points. 
Consequently, the hard threshold for simulation feasibility is an Out of Memory (OOM) error rather than an extended clock time.

This VRAM constraint directly impacts numerical stability. 
When simulations exhibit instability, increasing spatial resolution (grid fineness) is generally more effective than reducing the integration time step. 
While excessively large time steps will induce failures, diminishing returns are observed below a certain step-size threshold. 
This leaves grid resolution—and thereby VRAM capacity—as the primary lever for achieving stability.

* **Small test systems and simple polymer architectures:** A few minutes
* **Complex systems with stronger stability requirements:** Several minutes to a few hours
* **3D production systems:** Several hours to 2-3 days

Complex parameter sweeps used to map phase behavior scale efficiently by deploying independent simulation instances across multiple GPUs. 
For standard development and benchmarking in 2D, reasonable, well-parameterized simulations can be expected to complete within a single afternoon.

## Observables and Pressure evaluations

Computing the pressure observable requires propogating an additional derivative through the modified diffusion equation, and thus increases the cost of density computations by 
$25-50\%$ per cycle. As such, this operator should only be called when needed for optimal performance. 
