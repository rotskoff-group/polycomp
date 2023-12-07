import sys
#Needs to be changed to your specific path
sys.path.insert(0, "/home/emmit/Rotskoff/coacervation-dynamics/src/implicit_AB/soft_explicit/nanobrush")

from celluloid import Camera
import cupy as cp
import math
import matplotlib.pyplot as plt
import soft_exp_polymer as p
from mpl_toolkits.axes_grid1 import make_axes_locatable
from circle_function import draw_circle

cp.random.seed(1)

grid_spec = (230,230)
box_length = (16,16)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

nanp1, nanp1_brush = draw_circle(cp.array([12.5,4]), 2, grid)
nanp2, nanp2_brush = draw_circle(cp.array([12.5,12]), 2, grid)
nanp3, nanp3_brush = draw_circle(cp.array([4.5,4]), 2, grid)
nanp4, nanp4_brush = draw_circle(cp.array([4.5,12]), 2, grid)
A_brush = p.Brush('brush', nanp1_brush + nanp4_brush)
B_brush = p.Brush('brush', nanp2_brush + nanp3_brush)

A_mon = p.Monomer("A", 0.)
B_mon = p.Monomer("B", 0.)
C_mon = p.Monomer("C", 0)
S_mon = p.Monomer("S", 0)
N_mon = p.Monomer("N", 0)
monomers = [A_mon, B_mon, C_mon, N_mon, S_mon]

corr = 1
repul = 10.5
extra = 0
nano_pen = 0
FH_terms = {
        frozenset({A_mon}) : repul, 
        frozenset({B_mon}) : repul, 
        frozenset({C_mon}) : repul, 
        frozenset({S_mon}) : repul, 
        frozenset({N_mon}) : repul, 
        frozenset({A_mon, B_mon}) : repul + extra - 23, 
        frozenset({A_mon, C_mon}) : repul + extra, 
        frozenset({A_mon, S_mon}) : repul + extra, 
        frozenset({A_mon, N_mon}) : repul + extra + nano_pen, 
        frozenset({B_mon, C_mon}) : repul + extra, 
        frozenset({B_mon, S_mon}) : repul + extra, 
        frozenset({B_mon, N_mon}) : repul + extra + nano_pen,
        frozenset({C_mon, S_mon}) : repul + extra, 
        frozenset({C_mon, N_mon}) : repul + extra + nano_pen,
        frozenset({N_mon, S_mon}) : repul + extra + nano_pen, 
        }

N = 80
N2 = 80

for key in FH_terms.keys():
    FH_terms[key] = FH_terms[key] * 1

A_poly = p.Polymer("A", N, [(C_mon, .875 + 0), (A_mon, .125)], fastener=A_brush)
B_poly = p.Polymer("B", N2, [(C_mon, .875 - .0), (B_mon, .125)], fastener=B_brush)
polymers = [A_poly, B_poly]
spec_dict = {
        A_poly : .4 * 2,
        B_poly : .4 * 2,
        S_mon : 5 * N 
        }

nanp = p.Nanoparticle("N", N_mon, (nanp1 + nanp2 + nanp3 + nanp4) / (cp.average(nanp1[nanp1 > 0])\
        + cp.average(nanp2[nanp2 > 0]) + cp.average(nanp3[nanp3 > 0] + cp.average(nanp4[nanp4 > 0]))) * 200)
nanp_list = [nanp]
smear = 0.1632993161855452
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.020, integration_width = 1/7, nanoparticles = nanp_list)

ps.update_normal_from_density()
ps.update_density_from_normal()


relax_rates = cp.array([3 * 1 * corr * ps.grid.dV]*(ps.w_all.shape[0])) 
relax_temps = cp.array([0.001]*(ps.w_all.shape[0])) 
relax_temps *= ps.gamma.real
psi_rate = 20 * ps.grid.dV
print(psi_rate)
#psi_rate = 0.006096631 * N**2
#psi_rate = 20 * N / 500
psi_temp = 1
E = 350 
E = 700 
E = 4100
ps.update_normal_from_density()
ps.update_density_from_normal()
integrator_1 = p.CL_RK2(ps, relax_rates, relax_temps, psi_rate, psi_temp, E)

nrows=3
ncols=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=170, figsize=(4,6))
fig.suptitle('2 Nanoparticles, N-C-A and N-C-B structure')
multi_cam = Camera(fig)
ps.update_density_from_normal()
ps.update_normal_from_density()
ps.get_densities()



ps.update_density_from_normal()
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
im[0][0] = axes[0,0].imshow(ps.phi_all[ps.monomers.index(S_mon)].real.get(), cmap = 'Greens')
axes[0,0].set_title('Solvent Dens')
im[0][1] = axes[0,1].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
axes[0,1].set_title('A Dens')
im[1][0] = axes[1,0].imshow(ps.phi_all[ps.monomers.index(C_mon)].real.get(), cmap = 'Purples')
axes[1,0].set_title('Spacer Dens (cut)')
im[1][1] = axes[1,1].imshow(ps.phi_all[ps.monomers.index(B_mon)].real.get(), cmap = 'Reds')
axes[1,1].set_title('B Dens')
im[2][0] = axes[2,0].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
axes[2,0].set_title('Total density')
im[2][1] = axes[2,1].imshow(ps.phi_all[ps.monomers.index(N_mon)].real.get(), cmap = 'Oranges')
axes[2,1].set_title('Nanoparticle Dens')


steps = 200 * 30 // 30
for i in range(30):
    hold_S = cp.zeros_like(ps.phi_all[0].real)
    hold_A = cp.zeros_like(ps.phi_all[0].real)
    hold_B = cp.zeros_like(ps.phi_all[0].real)
    hold_C = cp.zeros_like(ps.phi_all[0].real)
    hold_T = cp.zeros_like(ps.phi_all[0].real)
    hold_N = cp.zeros_like(ps.phi_all[0].real)
    for _ in range(steps):
        integrator_1.ETD()
        hold_S += ps.phi_all[ps.monomers.index(S_mon)].real / steps
        hold_A += ps.phi_all[ps.monomers.index(A_mon)].real / steps
        hold_B += ps.phi_all[ps.monomers.index(B_mon)].real / steps
        hold_C += ps.phi_all[ps.monomers.index(C_mon)].real / steps
        hold_N += ps.phi_all[ps.monomers.index(N_mon)].real / steps
        hold_T += cp.sum(ps.phi_all, axis=0).real / steps
    charge_max = cp.amax(cp.abs(hold_C))
    hold_C[hold_C > 6] = 6
    im[0][0] = axes[0,0].imshow(hold_S.get(), cmap = 'Greens')
    im[0][1] = axes[0,1].imshow(hold_A.get(), cmap = 'Blues')
    im[1][0] = axes[1,0].imshow(hold_C.get(), cmap = 'Purples')
    im[1][1] = axes[1,1].imshow(hold_B.get(), cmap = 'Reds')
    im[2][0] = axes[2,0].imshow(hold_T.get(), cmap = 'Greys')
    im[2][1] = axes[2,1].imshow(hold_N.get(), cmap = 'Oranges')
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
plt.show()
