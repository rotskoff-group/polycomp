from celluloid import Camera
import cupy as cp
import math
import matplotlib.pyplot as plt
import soft_exp_polymer as p
from mpl_toolkits.axes_grid1 import make_axes_locatable

cp.random.seed(1)

grid1d = p.Grid(box_length=tuple([2]), grid_spec = tuple([20]))
grid2d = p.Grid(box_length=(2,2), grid_spec = (20,20))
grid3d = p.Grid(box_length=(2,2,2), grid_spec = (20,20,20))

print(grid1d.grid.shape)
print(grid2d.grid.shape)
print(grid3d.grid.shape)

print(grid1d.V)
print(grid2d.V)
print(grid3d.V)

A_mon = p.Monomer("A", 1)
B_mon = p.Monomer("B", -1)
monomers = [A_mon, B_mon]
#monomers = [C_mon, B_mon, A_mon]

FH_terms = {
        frozenset({A_mon}) : 0.05, 
        frozenset({B_mon}) : 0.05, 
        frozenset({A_mon, B_mon}) : 0.05, 
        }

N = 81 * 2 
N2 = 81

for key in FH_terms.keys():
    FH_terms[key] = FH_terms[key] 

Rg = 9
A_poly = p.Polymer("A", N, [(A_mon, 1)])
B_poly = p.Polymer("B", N2, [(B_mon, 1)])
AB_poly = p.Polymer("AB", N, [(A_mon, 0.5), (B_mon, 0.5)])
#ABC_poly = p.Polymer("ABC", N*3, [(A_mon, 0.15), (B_mon, 0.15), (C_mon, 0.7)])
polymers = [A_poly, B_poly]
polymers = [AB_poly]
spec_dict = {
        A_poly : 1.5,
        B_poly : 1.5 * N / N2,
        }
spec_dict = {
        AB_poly : 1.5 * 2
        }

grid_spec = (16,16,256)
box_length = (3 * 9,3 * 9,45 * 9)
box_length = (3,3,45)
#grid_spec = (760,760)

smear = 0.1632993161855452
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        box_length, grid_spec, smear, salt_conc=50 / N, integration_width = 1/(N))
#Test working polymer buildout
#ABC_poly.build_working_polymer(1)

#ps.w_all = ps.randomize_array(ps.w_all, .0001)
#cp.save('start_w', ps.w_all)
#ps.w_all = cp.load('start_w.npy')
ps.update_normal_from_density()
ps.update_density_from_normal()


#ps.get_densities()
#print(cp.average(ps.phi_all, axis=(1,2)))
#ps.update_normal_from_density()


relax_rates = cp.array([40 * ps.grid.dV]*(ps.w_all.shape[0])) 
relax_temps = cp.array([0.0000001]*(ps.w_all.shape[0]))  * 0
relax_temps *= ps.gamma.real
psi_rate = 35 * ps.grid.dV
#psi_rate = 0.006096631 * N**2
#psi_rate = 20 * N / 500
psi_temp = 1
E = 16000
#E = 1
#E = 219.4787379972565 * N**2 / Rg
ps.update_normal_from_density()
ps.update_density_from_normal()
integrator_1 = p.CL_RK2(ps, relax_rates, relax_temps, psi_rate, psi_temp, E)
#for _ in range(900):
#    integrator_1.ETD()

print(ps.normal_modes)
print(ps.normal_evalues)

nrows=3
ncols=2
fig, ax = plt.subplots()
ax2 = ax.twinx()
multi_cam = Camera(fig)
#cp.save("w_norm1.npy", ps.normal_w)
#ps.normal_w = cp.load("w_norm1.npy")
ps.update_density_from_normal()
ps.update_normal_from_density()
ps.get_densities()

#print(cp.average(ps.phi_all, axis=(1,2,3)))
#exit()


ps.psi[:,:,90:110] = 0
ps.w_all[0,:,:,50:150] = 0

##cp.save("w_dens1.npy", ps.w_all)
#ps.w_all = cp.load("w_dens1.npy")
ps.update_normal_from_density()
#ps.psi = cp.load('test_psi.npy') * N
ps.get_densities()
print(cp.average(ps.get_total_charge(),axis=(0,1)))
print(cp.average(ps.phi_all, axis=(1,2,3)))
#print(cp.average(ps.phi_salt, axis=(1,2,3)))

colors = ['r', 'b', 'k','g', 'orange', 'violet', 'plum', 'slategray', 'tab:brown']

ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_all[ps.monomers.index(A_mon)],
    axis=(0,1)).get().real,
    color=colors[0], label='real cation density')
ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_all[ps.monomers.index(B_mon)],
    axis=(0,1)).get().real,
    color=colors[1], label='real anion density')
ax.plot(ps.grid.grid[2,2,2].get(), cp.average(cp.sum(ps.phi_all,axis=0),axis=(0,1)).get().real,
        color=colors[2], label='total density')
ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.psi,axis=(0,1)).get().real,
        color=colors[3], label='psi')
ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.w_all[-1],axis=(0,1)).get().real,
        color=colors[5], label='w')
#ax2.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.psi,axis=(0,1)).get().imag,
#        color=colors[6], label='psi_imag')
ax2.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.gaussian_smear(ps.psi, smear),axis=(0,1)).get().imag,
        color=colors[6], label='psi_imag')
ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_salt[0],
    axis=(0,1)).get().real, color=colors[7], label='salt+ density')
ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_salt[1],
    axis=(0,1)).get().real, color=colors[8], label='salt- density')
#plt.show()
#exit()

for i in range(30):
    for _ in range(80):
        print(ps.psi[1,1,1])
        integrator_1.ETD()
        #integrator_1.remove_outliers()
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_all[ps.monomers.index(A_mon)],
        axis=(0,1)).get().real,
        color=colors[0], label='real cation density')
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_all[ps.monomers.index(B_mon)],
        axis=(0,1)).get().real,
        color=colors[1], label='real anion density')
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(cp.sum(ps.phi_all,axis=0),axis=(0,1)).get().real,
            color=colors[2], label='total density')
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.psi,axis=(0,1)).get().real,
            color=colors[3], label='psi')
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.w_all[-1],axis=(0,1)).get().real,
            color=colors[5], label='w')
    ax2.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.gaussian_smear(ps.psi, smear),axis=(0,1)).get().imag,
            color=colors[6], label='psi_imag')
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_salt[0],
        axis=(0,1)).get().real, color=colors[7], label='salt+ density')
    ax.plot(ps.grid.grid[2,2,2].get(), cp.average(ps.phi_salt[1],
        axis=(0,1)).get().real, color=colors[8], label='salt- density')
    multi_cam.snap()
    print(i)

#cp.save('test_psi', ps.psi)
#Random plotting garbage, sure there's a better way to do this
#for part in axes:
#    for ax in part:
#        ax.set_xticks([])
#        ax.set_yticks([])
#
#for i in range(nrows):
#    for j in range(ncols):
#        if im[i][j] is 0:
#            continue
#        if nrows==1:
#            div[i][j] = make_axes_locatable(axes[j])
#        else:
#            div[i][j] = make_axes_locatable(axes[i,j])
#        cax[i][j] = div[i][j].append_axes('right', size='8%', pad=0.02)
#for i in range(nrows):
#    for j in range(ncols):
#        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
#        if im[i][j] is 0:
#            continue
#        if nrows==1:
#            div[i][j] = make_axes_locatable(axes[j])
#        else:
#            div[i][j] = make_axes_locatable(axes[i,j])
#        cax[i][j] = div[i][j].append_axes('right', size='8%', pad=0.02)
#        cb[i][j].remove()
#        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
fig.tight_layout()
multimation = multi_cam.animate()
multimation.save('movie_traj.gif', writer='pillow')
plt.show()
