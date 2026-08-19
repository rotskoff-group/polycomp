import cupy as cp
import matplotlib.pyplot as plt
from celluloid import Camera
from mpl_toolkits.axes_grid1 import make_axes_locatable

import polycomp.ft_system as p
from polycomp.observables import get_free_energy

'''
This example provides a smaller, coarser system of the type computed in the original Pert 2025
coacervation paper. Full replication can be done by following the modifications specified in
methods section of that paper. Coacervation code is generally less stable, so this simplified version 
will still take an hour or more to run on most hardware, and the larger, finer simulations will 
take proportionally longer. 
'''
# Set a seed for reproducibility (or turn off for full randomization)
cp.random.seed(0)

# Declare all of your polymers with their name and charge
Lipid_mon = p.Monomer("Lipid", 0)
Cat_mon = p.Monomer("Cation", 1)
Ani_mon = p.Monomer("Anion", -1)
Solv_mon = p.Monomer("Solvent", 0)

# Declare a list of all the monomer types in the simulation
monomers = [Lipid_mon, Cat_mon, Ani_mon, Solv_mon]

tot = 1.5
lipid_hydrophobic = 4.0

# Declare a Flory-Huggins array with each cross iteraction and self-interaction
FH_terms = {
    frozenset({Lipid_mon}): tot,
    frozenset({Cat_mon}): tot,
    frozenset({Ani_mon}): tot,
    frozenset({Solv_mon}): tot,
    frozenset({Lipid_mon, Solv_mon}): tot + lipid_hydrophobic,
    frozenset({Cat_mon, Solv_mon}): tot,
    frozenset({Ani_mon, Solv_mon}): tot,
    frozenset({Lipid_mon, Cat_mon}): tot + lipid_hydrophobic,
    frozenset({Lipid_mon, Ani_mon}): tot + lipid_hydrophobic,
    frozenset({Cat_mon, Ani_mon}): tot,
}

# Declare the reference polymer length for the system.
N = 5

# Declare all the polymer types in solution.
Diblock = p.Polymer("Lipid-Cat", 2 * N, [(Lipid_mon, 0.5), (Cat_mon, 0.5)])
mRNA = p.Polymer("mRNA", 1 * N, [(Ani_mon, 1.0)])
Solvent = p.Polymer("Solvent", 1, [(Solv_mon, 1.0)])

# Declare a list of all the polymers in simulation
polymers = [Diblock, mRNA, Solvent]

# Declare a dictionary with all the species in the system and their concentrations
spec_dict = {
    Diblock: 0.3,
    mRNA: 0.3,
    Solvent: 1.5,
}

# Declare the number of grid points across each axis.
grid_spec = (128, 128)

# Declare the side length of the box along each axis.
box_length = (15, 15)

# Declare the grid object as specified using our parameters.
grid = p.Grid(box_length=box_length, grid_spec=grid_spec)

# Declare the smearing length for the charge and density
smear = 0.163

# We can now declare the full polymer system.
Cs = 0.0
ps = p.PolymerSystem(
    monomers,
    polymers,
    spec_dict,
    FH_terms,
    grid,
    smear,
    salt_conc=Cs * N,
    integration_width=1 / 20,
)

# Now we move to our integration parameters.
relax_rates = cp.array([0.0032] * (ps.w_all.shape[0])) * 2

# We also declare a temperature array which is the same shape
T_hot = 3e-3
T_cold = 1e-5

temps = cp.array([T_hot + 0j] * (ps.w_all.shape[0]))
temps *= ps.gamma.real

# Because this is a charged system, we set E and physical field rates
E = 1e4
psi_rate = 0.0080
psi_temp = 1.0

# Now we actually declare the integrator
integrator = p.CL_RK2(ps, relax_rates, temps, psi_rate, psi_temp, E)

# These are all plotting parameters
nrows = 1
ncols = 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170, figsize=(6, 2))
fig.suptitle("Coacervation Microphase Separation")
multi_cam = Camera(fig)

# generate an initial density for the starting plot
ps.get_densities()

im = []
div = []
cax = []
cb = []
for i in range(nrows):
    im.append([0] * ncols)
    div.append([0] * ncols)
    cax.append([0] * ncols)
    cb.append([0] * ncols)

# Initial plots
im[0][0] = axes[0].imshow(ps.phi_all[ps.monomers.index(Lipid_mon)].real.get(), cmap="Greens")
axes[0].set_title("Lipid Dens")
im[0][1] = axes[1].imshow(ps.phi_all[ps.monomers.index(Cat_mon)].real.get(), cmap="Blues")
axes[1].set_title("Cation Dens")
im[0][2] = axes[2].imshow(ps.phi_all[ps.monomers.index(Ani_mon)].real.get(), cmap="Reds")
axes[2].set_title("Anion Dens")
im[0][3] = axes[3].imshow(ps.phi_all[ps.monomers.index(Solv_mon)].real.get(), cmap="Purples")
axes[3].set_title("Solvent Dens")
im[0][4] = axes[4].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap="Greys")
axes[4].set_title("Total Dens")

# Declare some empty arrays to store our variables
dens_traj = []
free_energy_traj = []

# Set the number of steps per frame to average over the last 6000 steps of an interval
steps = 3000 

# Set the number of arrays to capture
for i in range(30):

    # Switch to colder temperature after 3.6 x 10^4 steps (i.e., after 6 loops)
    if i == 6:
        temps = cp.array([T_cold + 0j] * (ps.w_all.shape[0]))
        temps *= ps.gamma.real
        integrator = p.CL_RK2(ps, relax_rates, temps, psi_rate, psi_temp, E)

    hold_Lipid = cp.zeros_like(ps.phi_all[0].real)
    hold_Cat = cp.zeros_like(ps.phi_all[0].real)
    hold_Ani = cp.zeros_like(ps.phi_all[0].real)
    hold_Sol = cp.zeros_like(ps.phi_all[0].real)
    hold_Tot = cp.zeros_like(ps.phi_all[0].real)
    
    for _ in range(steps):

        # Collect the variables of interest every step and average over some of them
        free_energy_traj.append(get_free_energy(ps, E))
        integrator.ETD()
        hold_Lipid += ps.phi_all[ps.monomers.index(Lipid_mon)].real / steps
        hold_Cat += ps.phi_all[ps.monomers.index(Cat_mon)].real / steps
        hold_Ani += ps.phi_all[ps.monomers.index(Ani_mon)].real / steps
        hold_Sol += ps.phi_all[ps.monomers.index(Solv_mon)].real / steps
        hold_Tot += cp.sum(ps.phi_all, axis=0).real / steps

    # Save intermediate results here
    #    cp.save("free_energy_traj", cp.array(free_energy_traj))
    dens_traj.append((hold_Lipid, hold_Cat, hold_Ani))

    # Create new plots (use celluloid to make a simple animation)
    im[0][0] = axes[0].imshow(hold_Lipid.get(), cmap="Greens", vmin=0)
    im[0][1] = axes[1].imshow(hold_Cat.get(), cmap="Blues", vmin=0)
    im[0][2] = axes[2].imshow(hold_Ani.get(), cmap="Reds", vmin=0)
    im[0][3] = axes[3].imshow(hold_Sol.get(), cmap="Purples", vmin=0)
    im[0][4] = axes[4].imshow(hold_Tot.get(), cmap="Greys", vmin=0)
    multi_cam.snap()

    # Save data needed to restart simulation
    #    cp.save("midpoint", ps.w_all)
    #    cp.save("psi_midpoint", ps.psi)
    #    cp.save("live_dens_traj", cp.array(dens_traj))

    # Print current progress
    print(i)

# Generate arrays from the lists for easier handling
dens_traj = cp.array(dens_traj)
free_energy_traj = cp.array(free_energy_traj)

# Save the full trajectories
# cp.save("dens_traj", dens_traj)
# cp.save("free_energy_traj", free_energy_traj)

# The next section is all just to make relatively nice animations, mainly around
# accurately handling color bar scales so the 2D plots are interpretable
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

for i in range(nrows):
    for j in range(ncols):
        if im[i][j] == 0:
            continue
        if nrows == 1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i, j])
        cax[i][j] = div[i][j].append_axes("right", size="8%", pad=0.02)
for i in range(nrows):
    for j in range(ncols):
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation="vertical")
        if im[i][j] == 0:
            continue
        if nrows == 1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i, j])
        cax[i][j] = div[i][j].append_axes("right", size="8%", pad=0.02)
        cb[i][j].remove()
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation="vertical")

# Final plotting and saving the figures
fig.tight_layout()
multimation = multi_cam.animate()
multimation.save("movie_traj_CART.gif", writer="pillow")
plt.show()
