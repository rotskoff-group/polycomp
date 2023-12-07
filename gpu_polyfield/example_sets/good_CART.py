import sys
sys.path.insert(0, "/home/emmit/Rotskoff/coacervation-dynamics/src/implicit_AB/soft_explicit/nanobrush")
from celluloid import Camera
import cupy as cp
import math
import matplotlib.pyplot as plt
import soft_exp_polymer as p
from mpl_toolkits.axes_grid1 import make_axes_locatable

cp.random.seed(1)

grid_spec = (220,220)
grid_spec = (300,300)
box_length = (25,25)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

A_mon = p.Monomer("A", 3)
B_mon = p.Monomer("B", -3)
S_mon = p.Monomer("S", 0)
L_mon = p.Monomer("L", 0)
monomers = [A_mon, B_mon, L_mon, S_mon]
monomers = [L_mon, B_mon, A_mon, S_mon]

corr = 1
tot = 3 * 20
tot = 200
lip_pen = 2.4
FH_terms = {
        frozenset({A_mon}) : tot, 
        frozenset({B_mon}) : tot, 
        frozenset({S_mon}) : tot, 
        frozenset({L_mon}) : tot, 
        frozenset({A_mon, B_mon}) : tot, 
        frozenset({A_mon, S_mon}) : tot, 
        frozenset({B_mon, S_mon}) : tot, 
        frozenset({A_mon, L_mon}) : tot + lip_pen,
        frozenset({B_mon, L_mon}) : tot + lip_pen,
        frozenset({S_mon, L_mon}) : tot + lip_pen, 
        }

N = 20
N2 = 20

for key in FH_terms.keys():
    FH_terms[key] = FH_terms[key] * 1

A_poly = p.Polymer("A", N, [(A_mon, 1), (L_mon, 1)])
B_poly = p.Polymer("B", N2, [(B_mon, 1), (L_mon, 1)])
#B_poly = p.Polymer("B", N2, [(B_mon, 1)])
#AB_poly = p.Polymer("AB", N, [(A_mon, 0.5), (B_mon, 0.5)])
#ABC_poly = p.Polymer("ABC", N*3, [(A_mon, 0.15), (B_mon, 0.15), (C_mon, 0.7)])
polymers = [A_poly, B_poly]
#polymers = [AB_poly]
spec_dict = {
        A_poly : .25,
        B_poly : .25,
        S_mon : 9.0 * N 
        }
#spec_dict = {
#        A_poly : .0001 * 2,
#        B_poly : .0001 * 2,
#        S_mon : 1.5 * N 
#        }
#spec_dict = {
#        A_poly : .75 * 2,
#        B_poly : .75 * 2,
#        S_mon : .000001 * N 
#        }

smear = 0.1632993161855452
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0, integration_width = 1/10)

#Test working polymer buildout
#ABC_poly.build_working_polymer(1)
#ps.N = 40
#ps.w_all = ps.randomize_array(ps.w_all, .0001)
#cp.save('start_w', ps.w_all)
#ps.w_all = cp.load('start_w.npy')
ps.update_normal_from_density()
ps.update_density_from_normal()


#ps.get_densities()
#print(cp.average(ps.phi_all, axis=(1,2)))
#ps.update_normal_from_density()


relax_rates = cp.array([8 * 1 * corr * ps.grid.dV]*(ps.w_all.shape[0])) 
print(relax_rates)
#relax_rates[0] *= 2
relax_temps = cp.array([0.001]*(ps.w_all.shape[0])) 
relax_temps *= ps.gamma.real
psi_rate = 1.8 * ps.grid.dV
print(psi_rate)
#psi_rate = 0.006096631 * N**2
#psi_rate = 20 * N / 500
psi_temp = 1
E = 350 
E = 700 
E = 1000
E = 4100
E = 400
E = 1000
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
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170, figsize=(4,6))
fig.suptitle('50% CART 1:1 Lipid:Charge')
multi_cam = Camera(fig)
#cp.save("w_norm1.npy", ps.normal_w)
#ps.normal_w = cp.load("w_norm1.npy")
ps.update_density_from_normal()
ps.update_normal_from_density()
ps.get_densities()
ps.set_field_averages()

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
im[0][0] = axes[0,0].imshow(ps.phi_all[ps.monomers.index(S_mon)].real.get(), cmap = 'Greens')
axes[0,0].set_title('Solvent Dens')
im[1][0] = axes[1,0].imshow(ps.phi_all[ps.monomers.index(L_mon)].real.get(), cmap = 'Purples')
axes[1,0].set_title('Lipid Dens')
#im[1][0] = axes[1,0].imshow(cp.sum(ps.phi_salt, axis=0).real.get(), cmap = 'Oranges')
#axes[1,0].set_title('Salt Dens')
im[2][0] = axes[2,0].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
axes[2,0].set_title('Total density')
im[0][1] = axes[0,1].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
axes[0,1].set_title('A Dens')
im[1][1] = axes[1,1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap = 'Reds')
axes[1,1].set_title('B Dens')
#im[2][1] = axes[2,1].imshow(ps.get_total_charge().real.get(), cmap = 'bwr', vmin = -charge_max, vmax = charge_max)
#axes[2,1].set_title('Net Charge')
im[2][1] = axes[2,1].imshow(ps.gaussian_smear(ps.psi, smear).imag.get(), cmap = 'bwr')
axes[2,1].set_title('$\\phi$, Imaginary Component')
#plt.show()
#exit()

free_en = []
when = 0
steps = int(150 * 30 / 30)
for i in range(30):
    hold_S = cp.zeros_like(ps.phi_all[0].real)
    hold_A = cp.zeros_like(ps.phi_all[0].real)
    hold_B = cp.zeros_like(ps.phi_all[0].real)
    hold_T = cp.zeros_like(ps.phi_all[0].real)
    hold_L = cp.zeros_like(ps.phi_all[0].real)
    hold_C = cp.zeros_like(ps.phi_all[0].real)
    for _ in range(steps):
        integrator_1.ETD()
        free_en.append((when, ps.get_free_energy(E).get()))
        when += 1
        hold_S += ps.phi_all[ps.monomers.index(S_mon)].real / steps
        hold_A += ps.phi_all[ps.monomers.index(A_mon)].real / steps
        hold_B += ps.phi_all[ps.monomers.index(B_mon)].real / steps
        hold_L += ps.phi_all[ps.monomers.index(L_mon)].real / steps
        hold_T += cp.sum(ps.phi_all, axis=0).real / steps
        hold_C += ps.get_total_charge().real / steps
    charge_max = cp.amax(cp.abs(hold_C))
    im[0][0] = axes[0,0].imshow(hold_S.get(), cmap = 'Greens')
    im[1][0] = axes[1,0].imshow(hold_L.get(), cmap = 'Purples')
    im[2][0] = axes[2,0].imshow(hold_T.get(), cmap = 'Greys')
    im[0][1] = axes[0,1].imshow(hold_A.get(), cmap = 'Blues')
    im[1][1] = axes[1,1].imshow(hold_B.get(), cmap = 'Reds')
    im[2][1] = axes[2,1].imshow(hold_C.get(), cmap = 'bwr', vmin = -charge_max, vmax = charge_max)
#    im[0][0] = axes[0,0].imshow(ps.phi_all[ps.monomers.index(S_mon)].real.get(), cmap = 'Greens')
#    im[1][0] = axes[1,0].imshow(cp.sum(ps.phi_salt, axis=0).real.get(), cmap = 'Oranges')
#    im[2][0] = axes[2,0].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
#    im[0][1] = axes[0,1].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
#    im[1][1] = axes[1,1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap = 'Reds')
#    im[2][1] = axes[2,1].imshow(ps.get_total_charge().real.get(), cmap = 'bwr', vmin = -charge_max, vmax = charge_max)
    im[2][1] = axes[2,1].imshow(ps.gaussian_smear(ps.psi, smear).imag.get(), cmap = 'bwr')
    multi_cam.snap()
    cp.save('midpoint', ps.w_all)
    cp.save('midpoint_psi', ps.psi)
    #multi_cam.snap()
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
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
        if im[i][j] is 0:
            continue
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
plt.savefig('last_frame')

free_en_traj = cp.array(free_en)
fig_en, ax_en = plt.subplots()
ax_en.plot(free_en_traj[:,0].real.get(), free_en_traj[:,1].real.get())
plt.show()
