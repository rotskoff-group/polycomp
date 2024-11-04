from celluloid import Camera
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#DEAR GOD I HAVE NO IDEA WHY BUT TORCH MUST BE IMPORTED BEFORE CUPY
import torch
from polycomp.observables import * 
import cupy as cp
import copy
import polycomp.ft_system as p
from polycomp.observables import * 
from polycomp.neural_ansatz import * 

import torch
import sys

#Set a seed for reproducibility (or turn off for full randomization)
cp.random.seed(int(sys.argv[1]))

#if sys.argv[1] not in ['ABC', 'AB', 'AC', 'BC', 'ABCCBA', 'CAABBC']:
#    raise ValueError("wrong key")
#else:
#     poly_key = sys.argv[1]

#Declare all of your polymers with their name and charge
A_mon = p.Monomer("A", 0)
B_mon = p.Monomer("B", 0)
C_mon = p.Monomer("C", 0)

#Declare a list of all the monomer types in the simulation
#(Salts are handled automatically if needed)
monomers = [A_mon, B_mon, C_mon]

tot = 130
diff = 21

#Declare a Flory-Huggins array with each cross iteraction
#Here we use total and difference to simplify the description
FH_terms = {
        frozenset({A_mon}) : tot, 
        frozenset({B_mon}) : tot, 
        frozenset({C_mon}) : tot, 
        frozenset({A_mon, B_mon}) : tot + diff, 
        frozenset({A_mon, C_mon}) : tot + diff, 
        frozenset({B_mon, C_mon}) : tot + diff, 
        }

#Declare the reference polymer length for the system. In this case it will be the
# same as the length of the only polymer in solution 
N = 5

#Declare all the polymer types in solution. In this case we have a single "AB" diblock
# copolymer that is half A and half B, with a total length of N.
poly_options =  {
'ABC' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (B_mon, 1.0/3), (C_mon, 1.0/3)]), 1, 
         "run-20240627_002815-ABC1719473293.6887264"), 
'ABCCBA' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (B_mon, 1.0/3), (C_mon, 2.0/3), (B_mon, 1.0/3), (A_mon, 1.0/3)]), 2, 
           "run-20240627_162844-ABCCBA1719530923.2650678" ),
'CAABBC' : (p.Polymer("ABC", N, [(C_mon, 1.0/3), (A_mon, 2.0/3), (B_mon, 2.0/3), (C_mon, 1.0/3)]), 2, 
           "run-20240627_162828-CAABBC1719530907.5312178" ),
'AB' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (B_mon, 1.0/3)]), 2.0/3,
        "run-20240627_002813-AB1719473292.2740755"),
'AC' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (C_mon, 1.0/3)]), 2.0/3, 
        "run-20240627_002758-AC1719473276.9141884"),
'BC' : (p.Polymer("ABC", N, [(B_mon, 1.0/3), (C_mon, 1.0/3)]), 2.0/3,
        "run-20240627_004613-BC1719474372.3804157"),
}

ABC_poly = poly_options['ABC'][0]
ABCCBA_poly = poly_options['ABCCBA'][0]
CAABBC_poly = poly_options['CAABBC'][0]
AB_poly = poly_options['AB'][0]
AC_poly = poly_options['AC'][0]
BC_poly = poly_options['BC'][0]


#Declare a list of all the polymers in simulation 
polymers = [ABC_poly, ABCCBA_poly, CAABBC_poly, AB_poly, AC_poly, BC_poly]

rand_list = cp.sort(cp.random.rand(5))
norm_amt = {
        'ABC' : rand_list[0],
        'ABCCBA' : rand_list[1] - rand_list[0],
        'CAABBC' : rand_list[2] - rand_list[1],
        'AB' : rand_list[3] - rand_list[2],
        'AC' : rand_list[4] - rand_list[3],
        'BC' : 1 - rand_list[4],
        }

#tot = sum(norm_amt.values())
#for key in norm_amt.keys():
#    norm_amt[key] = norm_amt[key] / tot
if not math.isclose(sum(norm_amt.values()), 1.0):
    raise ValueError("Norm dict not sum to 1")
print(norm_amt)
#Declare a dictionary with all the species in the system (this will include polymers
# and solvents, but here we just have one polymer). We also declare the concentration
# of each species, in this case just 1. 
spec_dict = {
        ABC_poly : norm_amt['ABC'] / poly_options['ABC'][1],
        ABCCBA_poly : norm_amt['ABCCBA'] / poly_options['ABCCBA'][1],
        CAABBC_poly : norm_amt['CAABBC'] / poly_options['CAABBC'][1],
        AB_poly : norm_amt['AB'] / poly_options['AB'][1],
        AC_poly : norm_amt['AC'] / poly_options['AC'][1],
        BC_poly : norm_amt['BC'] / poly_options['BC'][1],
        }
#Declare the number of grid points across each axis. This will be a 2D simulation 
# with 256 grid points along each dimension. 
grid_spec = (256,256)

#Declare the side length of the box along each axis. Here we have 25x25 length square.
box_length = (16,16)

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
relax_rates = cp.array([1.0]*(ps.w_all.shape[0])) * 1.5

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
nrows=2
ncols=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170, figsize=(4,5))
formatted_title = "ABC : {:.3f}, ABCCBA : {:.3f}, CAABBC : {:.3f}, \n AB : {:.3f}, AC : {:.3f}, BC : {:.3f}".format(norm_amt['ABC'], 
                                                                                                             norm_amt['ABCCBA'],
                                                                                                             norm_amt['CAABBC'],
                                                                                                             norm_amt['AB'],
                                                                                                             norm_amt['AC'],
                                                                                                             norm_amt['BC'])

fig.suptitle(formatted_title)
multi_cam = Camera(fig)

#generate an initial density for the starting plot
integrator.ETD()
model_dict = copy.copy(ps.poly_dict)
model_dir = "/scratch/users/epert/ABC_neural/production_ABC/wandb_logs/"
model_dict[ABC_poly] = model_dir + poly_options['ABC'][2] + "/checkpoints/"
model_dict[ABCCBA_poly] = model_dir + poly_options['ABCCBA'][2] + "/checkpoints/"
model_dict[CAABBC_poly] = model_dir + poly_options['CAABBC'][2] + "/checkpoints/"
model_dict[AB_poly] = model_dir + poly_options['AB'][2] + "/checkpoints/"
model_dict[AC_poly] = model_dir + poly_options['AC'][2] + "/checkpoints/"
model_dict[BC_poly] = model_dir + poly_options['BC'][2] + "/checkpoints/"
model = create_ansatz(ps, model_dict)
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
im[0][0] = axes[0][0].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
axes[0][0].set_title('A Dens')
im[0][1] = axes[0][1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap = 'Reds')
axes[0][1].set_title('B Dens')
im[1][0] = axes[1][0].imshow(ps.phi_all[ps.monomers.index(C_mon)].real.get(), cmap = 'Greens')
axes[1][0].set_title('C Dens')
im[1][1] = axes[1][1].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
axes[1][1].set_title('Total density')

#Declare some empty arrays to store our variables
dens_traj = []
free_energy_traj = []

#Set the number of steps per frame
steps = 6000

f_traj = []
d_traj = []
error_traj = []
abs_error_traj = []
#Set the number of arrays to capture
import time
t0 = time.time()
sampling_frequency = 50
for i in range(1):

    #We average over multiple views to reduce the noise for visualization, could just 
    # plot directly as well to simplify this
#    ps.w_all.real = cp.random.normal(loc=0.0, scale=0.1, size=ps.w_all.shape) 
    ps.w_all.imag *= 0
#    ps.w_all[ps.w_all.real < 0] = -5
#    ps.w_all[ps.w_all.real > 0] = 5
    hold_A = cp.zeros_like(ps.phi_all[0].real)
    hold_B = cp.zeros_like(ps.phi_all[0].real)
    hold_C = cp.zeros_like(ps.phi_all[0].real)
    hold_f = cp.zeros_like(ps.phi_all[0].real)
    hold_T = cp.zeros_like(ps.phi_all[0].real)
    for _ in range(steps):
        if _ % 500==0:
            print(_)

        #Collect the variables of interest every step and average over some of them
        free_energy_traj.append(get_free_energy(ps, E))
        if _ % sampling_frequency == 0:
            #infer(ps)
            ps.get_densities()
            correct_vals = cp.copy(ps.phi_all.real)
        integrator.ETD(neural_ansatz=True)
        if _ % sampling_frequency == 0:
            test_vals = ps.phi_all.real
            rel_error = cp.average(cp.abs(test_vals - correct_vals) / cp.abs(correct_vals)).get()
            abs_error = cp.average(cp.abs(test_vals - correct_vals)).get()
            error_traj.append(rel_error)
            abs_error_traj.append(abs_error)
        hold_A = ps.phi_all[ps.monomers.index(A_mon)].real #/ steps
        hold_B = ps.phi_all[ps.monomers.index(B_mon)].real #/ steps
        hold_C = ps.phi_all[ps.monomers.index(C_mon)].real #/ steps
        hold_f = ps.w_all[0].real #/ steps
        hold_T = cp.sum(ps.phi_all, axis=0).real #/ steps
        if _ % (steps // 30) == 0 :
            im[0][0] = axes[0][0].imshow(hold_A.get(), cmap = 'Blues', vmin = 0)
            im[0][1] = axes[0][1].imshow(hold_B.get(), cmap = 'Reds', vmin = 0)
            im[1][0] = axes[1][0].imshow(hold_C.get(), cmap = 'Greens', vmin = 0)
            im[1][1] = axes[1][1].imshow(hold_T.get(), cmap = 'Greys', vmin = 0)
            dens_traj.append((hold_A, hold_B, hold_C))
            multi_cam.snap()
#        if (_+1) % 200 == 0 :
#            d_traj.append(cp.stack((cp.copy(ps.phi_all[ps.monomers.index(A_mon)]), cp.copy(ps.phi_all[ps.monomers.index(B_mon)])), axis=0))
    print(i)
    print(time.time() - t0)
    t0 = time.time()

    #Save intermediate results here
#    t1 = time.time()
#    cp.save('free_energy_traj', cp.array(free_energy_traj))
#    print(time.time()- t1)
#    dens_traj.append((hold_A, hold_f, hold_T))

    #Create new plots (use celluloid to make a simple animation)
#    multimation = multi_cam.animate()
#    plt.show()
#    exit()
#    im[0][0] = axes[0].imshow(hold_A.get(), cmap = 'Blues', vmin = 0)
#    im[0][1] = axes[1].imshow(hold_f.get(), cmap = 'Reds') #vmin = 0)
#    im[0][2] = axes[2].imshow(hold_T.get(), cmap = 'Greys', vmin = 0)
#    plt.show()
#    exit()
#    multi_cam.snap()

    #Save data needed to restart simulation
    cp.save('midpoint', ps.w_all)
    cp.save('psi_midpoint', ps.psi)
#    cp.save('live_dens_traj', cp.array(dens_traj))

    #Print current progress
#    if i % 1 ==0: 
#        print(time.time() - t0)
#        t0 = time.time()
#        print(i)

#Generate arrays from the lists for easier handling 
dens_traj = cp.array(dens_traj)
free_energy_traj = cp.array(free_energy_traj)

#Save the full trajectories
cp.save('dens_traj', dens_traj)
cp.save('free_energy_traj', free_energy_traj)
cp.save('error_traj', error_traj)
cp.save('abs_error_traj', abs_error_traj)


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
multimation.save('movie_traj_' + sys.argv[1] + '.gif', writer='pillow')
fig.savefig("last_frame_" + sys.argv[1] + ".pdf")
fig2= plt.figure()
ax = fig2.add_subplot(111)
ax.plot(error_traj)
ax.set_xlabel("Step")
ax.set_ylabel("Relative Error")
fig2.savefig("rel_error_" + sys.argv[1] + ".png")
#plt.show()

#torch.save(output, 'ABA_size1.pt')
exit()
