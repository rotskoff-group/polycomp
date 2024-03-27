from celluloid import Camera
import cupy as cp
import matplotlib.pyplot as plt
import polycomp.ft_system as p
from polycomp.observables import * 
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Set a seed for reproducibility (or turn off for full randomization)
cp.random.seed(0)

#Declare all of your polymers with their name and charge
A_mon = p.Monomer("A", 0)
B_mon = p.Monomer("B", 0)

#Declare a list of all the monomer types in the simulation
#(Salts are handled automatically if needed)
monomers = [A_mon, B_mon]

tot = 130
diff = .5

#Declare a Flory-Huggins array with each cross iteraction
#Here we use total and difference to simplify the description
FH_terms = {
        frozenset({A_mon}) : tot, 
        frozenset({B_mon}) : tot, 
        frozenset({A_mon, B_mon}) : tot + diff, 
        }

#Declare the reference polymer length for the system. In this case it will be the
# same as the length of the only polymer in solution 
N = 5

#Declare all the polymer types in solution. In this case we have a single "AB" diblock
# copolymer that is half A and half B, with a total length of N. 
A_poly = p.Polymer("A", N, [(A_mon, 1.0)])
B_poly = p.Polymer("B", N, [(B_mon, 1.0)])

#Declare a list of all the polymers in simulation 
polymers = [A_poly,B_poly]

#Declare a dictionary with all the species in the system (this will include polymers
# and solvents, but here we just have one polymer). We also declare the concentration
# of each species, in this case just 1. 
spec_dict = {
        A_poly : 1.,
        B_poly : 1.
        }
#Declare the number of grid points across each axis. This will be a 2D simulation 
# with 256 grid points along each dimension. 
grid_spec = (256,256)

#Declare the side length of the box along each axis. Here we have 25x25 length square.
box_length = (25,25)

#Declare the grid object as specified using our parameterss.
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

#Declare the smearing length for the charge and density
smear = 0.2

#We can now declare the full polymer system. Read the full documentation for details, 
# but we use previously declared variables and specify salt concentration and 
# integration fineness along the polymer. 
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0 * N, integration_width = 1/20)

#Now we move to our integration parameters. We need a timestep associated with each 
# field, but they'll all be the same here. 
relax_rates = cp.array([0.45]*(ps.w_all.shape[0]))

#We also declare a temperature array which is the same shape
temps = cp.array([0.001 + 0j]*(ps.w_all.shape[0]))

#This temperature corresponds to using the "standard" CL integrator, but other versions
# generally are valid
temps *= ps.gamma.real

#Because this is an uncharged system, we set E to 0, and set all the electric field 
# rates to 0 as well
E = 0
psi_rate = 0
psi_temp = 0

#Now we actually declare the integrator
integrator = p.CL_RK2(ps, relax_rates, temps, psi_rate, psi_temp, E)

#These are all plotting parameters and will need to be changed if we want more or less 
# plots
nrows=1
ncols=3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170, figsize=(6,2))
fig.suptitle('Example of simple microphase separation')
multi_cam = Camera(fig)

#generate an initial density for the starting plot
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

#Initial plots
im[0][0] = axes[0].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
axes[0].set_title('A Dens')
im[0][1] = axes[1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap = 'Reds')
axes[1].set_title('B Dens')
im[0][2] = axes[2].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
axes[2].set_title('Total density')

#Declare some empty arrays to store our variables
dens_traj = []
free_energy_traj = []

#Set the number of steps per frame
steps = 100

#Set the number of arrays to capture
for i in range(30):

    #We average over multiple views to reduce the noise for visualization, could just 
    # plot directly as well to simplify this
    hold_A = cp.zeros_like(ps.phi_all[0].real)
    hold_B = cp.zeros_like(ps.phi_all[0].real)
    hold_T = cp.zeros_like(ps.phi_all[0].real)
    for _ in range(steps):

        #Collect the variables of interest every step and average over some of them
        free_energy_traj.append(get_free_energy(ps, E))
        integrator.ETD()
        hold_A += ps.phi_all[ps.monomers.index(A_mon)].real / steps
        hold_B += ps.phi_all[ps.monomers.index(B_mon)].real / steps
        hold_T += cp.sum(ps.phi_all, axis=0).real / steps

    #Save intermediate results here
    cp.save('free_energy_traj', cp.array(free_energy_traj))
    dens_traj.append((hold_A, hold_B, hold_T))

    #Create new plots (use celluloid to make a simple animation)
    im[0][0] = axes[0].imshow(hold_A.get(), cmap = 'Blues', vmin = 0)
    im[0][1] = axes[1].imshow(hold_B.get(), cmap = 'Reds', vmin = 0)
    im[0][2] = axes[2].imshow(hold_T.get(), cmap = 'Greys', vmin = 0)
    multi_cam.snap()

    #Save data needed to restart simulation
    cp.save('midpoint', ps.w_all)
    cp.save('psi_midpoint', ps.psi)
    cp.save('live_dens_traj', cp.array(dens_traj))

    #Print current progress
    print(i)

#Generate arrays from the lists for easier handling 
dens_traj = cp.array(dens_traj)
free_energy_traj = cp.array(free_energy_traj)

#Save the full trajectories
cp.save('dens_traj', dens_traj)
cp.save('free_energy_traj', free_energy_traj)



#The next section is all just to make relatively nice animations, mainly around 
# accurately handling color bar scales so the 2D plots are interpretable 
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

for i in range(nrows):
    for j in range(ncols):
        if im[i][j]==0:
            continue
        if nrows==1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i,j])
        cax[i][j] = div[i][j].append_axes('right', size='8%', pad=0.02)
for i in range(nrows):
    for j in range(ncols):
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
        if im[i][j]==0:
            continue
        if nrows==1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i,j])
        cax[i][j] = div[i][j].append_axes('right', size='8%', pad=0.02)
        cb[i][j].remove()
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')

#Final plotting and saving the figures
fig.tight_layout()
multimation = multi_cam.animate()
multimation.save('movie_traj.gif', writer='pillow')
plt.show()

