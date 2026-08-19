import cupy as cp
import matplotlib.pyplot as plt
from celluloid import Camera
from mpl_toolkits.axes_grid1 import make_axes_locatable

import polycomp.ft_system as p
from polycomp.observables import get_free_energy

'''
This code produces a 2D simulation of a solid wall with attached brushed polymer on half of the 
wall. Generating other types of nanoparticle/fixed polymer simuations can be directly derived 
by changing the masks [brush_density/np_density] to the desired states, as well as standard
modifications to the polymers themselves. 
'''

# Set a seed for reproducibility
cp.random.seed(0)

# Declare all of your polymers and monomers with their name and charge
A_mon = p.Monomer("A", 0)
B_mon = p.Monomer("B", 0)
S_mon = p.Monomer("S", 0)
NP_mon = p.Monomer("NP", 0)

# Declare a list of all the monomer types in the simulation
monomers = [A_mon, B_mon, S_mon, NP_mon]

tot = 130 
diff = 2.0 

# Declare a Flory-Huggins array with each cross interaction and self-interaction
# The NP is treated as inert/repulsive to strongly phase separate it
FH_terms = {
    # Self-interactions
    frozenset({A_mon}): tot,
    frozenset({B_mon}): tot,
    frozenset({S_mon}): tot,
    frozenset({NP_mon}): tot,
    
    # Cross-interactions
    frozenset({A_mon, B_mon}): tot + diff,
    frozenset({A_mon, S_mon}): tot + diff,
    frozenset({B_mon, S_mon}): tot + diff,
    frozenset({A_mon, NP_mon}): tot + diff,
    frozenset({B_mon, NP_mon}): tot + diff,
    frozenset({S_mon, NP_mon}): tot + diff,
}

# Declare the reference polymer length for the system
N = 5

# Declare the grid and box size matching the diblock case
grid_spec = (256, 256)
box_length = (30, 30)

# Setup the Nanoparticle and Brush regions
np_density = cp.zeros(grid_spec)
# Fill the left 1/8th of the box (256 / 8 = 32)
np_density[:, :64] = 1.  # High density inert NP region

brush_density = cp.zeros(grid_spec)
# Attach the polymer to the cells immediately right of the NP region
brush_density[64:192, 64] = 3.0 

wall_np = p.Nanoparticle("Wall_NP", NP_mon, np_density)
wall_brush = p.Brush("Wall_Brush", brush_density)

# Declare all the polymer types in solution and attach with the fastener
AB_poly = p.Polymer("AB", N, [(A_mon, 1.5), (B_mon, 1.5)], fastener=wall_brush)

# Declare a list of all the polymers in simulation
polymers = [AB_poly]

# Declare a dictionary with all the species in the system
spec_dict = {AB_poly: 0.05, S_mon: 0.6 * N}

# Declare the grid object
grid = p.Grid(box_length=box_length, grid_spec=grid_spec)

# Declare the smearing length for the charge and density
smear = 0.2

# Declare the full polymer system, now including the nanoparticle wall
ps = p.PolymerSystem(
    monomers,
    polymers,
    spec_dict,
    FH_terms,
    grid,
    smear,
    salt_conc=0.0 * N,
    integration_width=1 / 20,
    nanoparticles=[wall_np]
)

# Integration parameters
relax_rates = cp.array([0.03] * (ps.w_all.shape[0]))
temps = cp.array([0.01 + 0j] * (ps.w_all.shape[0]))
temps *= ps.gamma.real

E = 0
psi_rate = 0
psi_temp = 0

# Declare the integrator
integrator = p.CL_RK2(ps, relax_rates, temps, psi_rate, psi_temp, E)

# Plotting parameters
nrows = 2
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170)
fig.suptitle("Density profile of lamellar diblock bound to NP Wall with solvent")
multi_cam = Camera(fig)

# Generate an initial density for the starting plot
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
im[0][0] = axes[0, 0].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap="Blues")
axes[0, 0].set_title("A Dens")
im[0][1] = axes[0, 1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap="Reds")
axes[0, 1].set_title("B Dens")
im[1][0] = axes[1, 0].imshow(ps.phi_all[ps.monomers.index(S_mon)].real.get(), cmap="Greens")
axes[1, 0].set_title("S Dens")
im[1][1] = axes[1, 1].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap="Greys")
axes[1, 1].set_title("Total density")

# Declare empty arrays to store trajectory variables
dens_traj = []
free_energy_traj = []

# Set the number of steps per frame
steps = 200

# Set the number of frames to capture
for i in range(40):

    hold_A = cp.zeros_like(ps.phi_all[0].real)
    hold_B = cp.zeros_like(ps.phi_all[0].real)
    hold_S = cp.zeros_like(ps.phi_all[0].real)
    hold_T = cp.zeros_like(ps.phi_all[0].real)
    
    for _ in range(steps):
        # Collect the variables we want every step and average over some of them
        free_energy_traj.append(get_free_energy(ps, E))
        integrator.ETD()
        hold_A += ps.phi_all[ps.monomers.index(A_mon)].real / steps
        hold_B += ps.phi_all[ps.monomers.index(B_mon)].real / steps
        hold_S += ps.phi_all[ps.monomers.index(S_mon)].real / steps
        hold_T += cp.sum(ps.phi_all, axis=0).real / steps

    # Save intermediate results here
    # cp.save("free_energy_traj", cp.array(free_energy_traj))
    dens_traj.append((hold_A, hold_B, hold_T))

    # Create new plots
    im[0][0] = axes[0, 0].imshow(hold_A.get(), cmap="Blues", vmin=0)
    im[0][1] = axes[0, 1].imshow(hold_B.get(), cmap="Reds", vmin=0)
    im[1][0] = axes[1, 0].imshow(hold_S.get(), cmap="Greens", vmin=0)
    im[1][1] = axes[1, 1].imshow(hold_T.get(), cmap="Greys", vmin=0)
    multi_cam.snap()

    # Print current progress
    print(i)

# Generate arrays from the lists for easier handling
dens_traj = cp.array(dens_traj)
free_energy_traj = cp.array(free_energy_traj)

# Setup color bars and formatting
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
multimation.save("movie_traj_wall.gif", writer="pillow")
plt.show()
