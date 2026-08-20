# Common Issues and Troubleshooting

Complex Langevin simulations of fluctuating polymer field theories are highly sensitive to their integration parameters. Below are common numerical and physical issues encountered during simulation and their respective solutions.

## `Q_c not equal across integral` Error

This is the most common error that you will encounter, and it will immediately kill the simulation. 
When computing the partition function, the value of the partition function should be exactly equal for every point along the polymer backbone. 
If this is not the case, the system cannot guarantee that the densities have any physical meaning, and thus proceeding with further integration is invalid. 
The underlying cause of this error comes in a few common flavors. 

* **Unstable Simulation:** The simulation was unstable. Plotting the density before the crash will show massive, unphysical behavior. 
These bad values eventually cause the integrator of the partition function to break in some way, resulting in a crash. 
* **Extremely sharp boundaries:** Simulations that develop extremely sharp boundaries will sometimes be relatively stable in density but still unable to maintain 
the required stability of the integrator if there are massive, sharp fluctuations in density. 
* **Issues with MDE protocol:** More common if modifying the MDE algorithm by hand, many small errors can cause the partition function to be incorrectly valued at 
one or more points along the backbone. In this case directly examining the values of the partition function to see how many are wrong can be helpful in diagnosing what the issue is. 
This also could indicate there is an underlying bug for the specific simulation type that you are running. 

!!! important "Do not suppress MDE errors"
    Do not suppress the MDE error. It is telling you something is severely wrong in the system. There are some cases where this type of MDE error will diverge and then reconverge
    later in the simulation, but this is extremely rare and associated only with very large, particular simulation regimes. 
    Any configuration where the value of `Q_c` is not equal at all points is invalid for all observables. 

## Integrable but unphysical simulations

Sometimes the simulation will complete successfully, but the solved equilibrium state will show near-singularities, rapid fluctuations, or other obviously incorrect behavior. 
This includes single points of very high density, often travelling across the simulation, or other anomalous behavior. 
In essentially all cases this is the result of unstable simulation parameters. 
One helpful thing to check when debugging this error is whether the behavior is oscillatory or not. 
Oscillatory behavior is usually the result of traditional overshooting integration and is often fixed by reducing the timestep. Non-oscillatory behavior often cannot be resolved
by changing the time step and will require changing other integration parameters. 
These two different failure modes can be distinguished by printing out the density at odd intervals and looking for rapid oscillations. Even interval printouts often mask the difference. 

## Improving Simulation Stability

Not all combinations of FH parameters are integrable. If the FH matrix is net attractive, the system will consistently form a singularity. 
Some other simulation parameters that are theoretically stable are not numerically stable in practice. If your simulation is numerically unstable try the following modifications:

* **Reduce or remove noise:** If the model does not strictly require noise for sampling, ensure that it is stable without any noise before trying to debug further. 
* **Increase grid fineness:** Either by increasing the number of grid points, or by reducing the side length of the grid. Empirically it appears to 
more often fix stability issues than changing the integration timestep if the timestep is known to not be implausibly large. 
* **Reduce integration time step:** Often fixes oscillatory type errors, also attributable to other errors as well. Testing out a few time scales and finding the stability boundary
is generally good practice. 
* **Increase fineness of polymer gridding:** Similar to increasing grid fineness and decreasing integration time. 
Empirically coarser polymers will be stable but have issues with the density profile itself, though sometimes this can improve performance. 
* **Modify FH matrix:** The FH matrix greatly affects the stability of the simulation. In general, large FH values and large differences in FH values cause numeric instability.
Reducing both is a good way to determine whether the issue is a challenging parameter regime (which may require more careful or higher resource integration elsewhere) or something
more profound. This will change the physics of the system but is useful in debugging. 
* **Increase smearing constants:** Also will physically change the system, but useful for debugging or for getting information about particularly stubborn systems. 
* **Modify charge behavior:** When charge is included, it can cause the simulation to become much more fragile, particularly if coacervation is occurring.
In this regime, changing charge behavior in addition to all previous approaches may be necessary to get a stable simulation. 

## Out of Memory (OOM) Errors

Because spectral methods require complete arrays to reside in GPU memory, highly resolved 3D grids can quickly exceed available VRAM. 
This error will generally limit the maximum size of simulation possible on your hardware. 

The largest arrays that will need to be declared for any simulation is the two `q_r_s`/`q_r_s_dag` arrays that need to be stored during the computation of the MDE. 
The size of these arrays scales as (number of grid points) * (number of non-degenerate species types) * (number of integration points along length of polymer). 
As such both of the following are helpful in reducing memory load. 

*   **Reduce Grid Dimensions:** Decrease the number of grid points per axis. These are also stored in other moderately large arrays that scale like grid size.
*   **Reduce number of integration points along polymer:** Use fewer integration points along the polymer. 

Both of these reduce system fidelity, but will generally set the upper bound of simulations that are possible to be conducted as there is no way to access more VRAM 
without changing hardware. High VRAM GPUs are useful when particularly large simulations are required. 

