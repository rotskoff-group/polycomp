import sys
sys.path.insert(0, "/home/emmit/Rotskoff/coacervation-dynamics/src/implicit_AB/soft_explicit/nanobrush")
from celluloid import Camera
import cupy as cp
import math
import matplotlib.pyplot as plt
import soft_exp_polymer as p
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys


A_amt = float(sys.argv[1])
cp.random.seed(int(A_amt * 1000000))

grid_spec = (230,230)
#grid_spec = (300,300)
box_length = (25,25)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

A_mon = p.Monomer("A", 0.)
B_mon = p.Monomer("B", 0.)
monomers = [A_mon, B_mon]

N = 2
N2 = 20

corr = 1
total = 10
diff = 2.5 * N
FH_terms = {
        frozenset({A_mon}) : total, 
        frozenset({B_mon}) : total, 
        frozenset({A_mon, B_mon}) : total + diff, 
        }


for key in FH_terms.keys():
    FH_terms[key] = FH_terms[key] * 1
total = 1

A_poly = p.Polymer("A", N, [(A_mon, 1), (A_mon,1)])
#B_poly = p.Polymer("B", N2, [(B_mon, 1)])
#B_poly = p.Polymer("B", N2, [(B_mon, 1)])
#AB_poly = p.Polymer("AB", N, [(A_mon, 0.5), (B_mon, 0.5)])
#ABC_poly = p.Polymer("ABC", N*3, [(A_mon, 0.15), (B_mon, 0.15), (C_mon, 0.7)])
polymers = [A_poly]
#polymers = [AB_poly]
spec_dict = {
        A_poly : A_amt,
        B_mon : (1-A_amt) * N
        }

smear = 0.1632993161855452
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0, integration_width = 1/5)

relax_rates = cp.array([20 * 2 * corr * ps.grid.dV]*(ps.w_all.shape[0])) 
print(relax_rates)
relax_temps = cp.array([0.001]*(ps.w_all.shape[0])) 
relax_temps *= ps.gamma.real
psi_rate = 20 * ps.grid.dV
print(psi_rate)
psi_temp = 1 * 0
E = 350 
E = 700 
E = 4100
ps.update_normal_from_density()
ps.update_density_from_normal()
integrator_1 = p.CL_RK2(ps, relax_rates, relax_temps, psi_rate, psi_temp, E)

nrows=2
ncols=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170, figsize=(4,6))
#fig.suptitle('A + B homopolymers')
print(psi_rate)
multi_cam = Camera(fig)
#cp.save("w_norm1.npy", ps.normal_w)
#ps.normal_w = cp.load("w_norm1.npy")
ps.update_density_from_normal()
ps.update_normal_from_density()
ps.get_densities()
ps.set_field_averages()
ps.get_chemical_potential()
#print(cp.average(ps.phi_all, axis=(1,2,3)))
#exit()



ps.update_density_from_normal()
##cp.save("w_dens1.npy", ps.w_all)
#ps.w_all = cp.load("midpoint.npy")
#ps.update_normal_from_density()
#ps.normal_w[-2].real = ps.randomize_array(ps.w_all[0], 30)
#ps.update_density_from_normal()
#ps.psi = cp.load('test_psi.npy') * N
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


psi_max = cp.amax(cp.abs(ps.psi.real))
charge_max = cp.amax(cp.abs(ps.get_total_charge().real))
print(im)
im[0][0] = axes[0,0].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
axes[0,0].set_title('A Dens')
im[0][1] = axes[0,1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap = 'Reds')
axes[0,1].set_title('B Dens')
im[1][0] = axes[1,0].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
axes[1,0].set_title('Total density')
#im[2][1] = axes[2,1].imshow(ps.get_total_charge().real.get(), cmap = 'bwr', vmin = -charge_max, vmax = charge_max)
#axes[2,1].set_title('Net Charge')
#plt.show()
#exit()

free_en = []
net_mu = []
net_pi = []
when = 0
steps = int(100 * 30 / 30)
for i in range(30):
    hold_A = cp.zeros_like(ps.phi_all[0].real)
    hold_B = cp.zeros_like(ps.phi_all[0].real)
    hold_T = cp.zeros_like(ps.phi_all[0].real)
    for _ in range(steps):
        integrator_1.ETD()
        free_en.append((when, ps.get_free_energy(E).get()))
        ps.get_chemical_potential()
        #net_mu.append((ps.chem_pot_dict[A_poly] - N * ps.chem_pot_dict[B_mon]))
        net_mu.append((ps.chem_pot_dict[A_poly], ps.chem_pot_dict[B_mon]))
        net_pi.append((ps.get_pressure()))
        when += 1
        hold_A += ps.phi_all[ps.monomers.index(A_mon)].real / steps
        hold_B += ps.phi_all[ps.monomers.index(B_mon)].real / steps
        hold_T += cp.sum(ps.phi_all, axis=0).real / steps
    im[0][0] = axes[0,0].imshow(hold_A.get(), cmap = 'Blues')
    im[0][1] = axes[0,1].imshow(hold_B.get(), cmap = 'Reds')
    im[1][0] = axes[1,0].imshow(hold_T.get(), cmap = 'Greys')
    multi_cam.snap()
    cp.save('midpoint', ps.w_all)
    cp.save('midpoint_psi', ps.psi)
    print(i)




#Random plotting garbage, sure there's a better way to do this
for part in axes:
    for ax in part:
        ax.set_xticks([])
        ax.set_yticks([])

for i in range(nrows):
    for j in range(ncols):
        if im[i][j] is 0:
            continue
        if nrows==1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i,j])
        cax[i][j] = div[i][j].append_axes('right', size='8%', pad=0.02)
for i in range(nrows):
    for j in range(ncols):
        if im[i][j] is 0:
            continue
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
        if nrows==1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i,j])
        cax[i][j] = div[i][j].append_axes('right', size='8%', pad=0.02)
        cb[i][j].remove()
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
fig.tight_layout()
multimation = multi_cam.animate()
multimation.save('movie_traj.gif', writer='pillow')
#plt.savefig('last_frame')

free_en_traj = cp.array(free_en)
#net_mu_traj = cp.expand_dims(cp.array(net_mu), axis=1)
net_mu_traj = cp.array(net_mu)
net_pi_traj = cp.expand_dims(cp.array(net_pi), axis=1)
print(net_pi_traj.shape)
full_traj = cp.concatenate((free_en_traj, net_mu_traj, net_pi_traj), axis=1)
fig_en, ax_en = plt.subplots()
ax_en.plot(free_en_traj[:,0].real.get(), free_en_traj[:,1].real.get())

#cp.savetxt("AS_traj/traj_%.5f" % A_amt, full_traj)
print("AS_traj/traj_%.5f" % A_amt)
plt.show()

