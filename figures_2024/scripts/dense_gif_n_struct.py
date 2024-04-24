import os
import sys
sys.path.insert(0, "/scratch/users/epert/nice_plotting/salt_tests/nanobrush")
from celluloid import Camera
import cupy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.colors as mcolors
import polycomp.ft_system as p
from polycomp.observables import get_structure_factor
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("~/Rotskoff/polycomp/figures_2024/stylefile.mplstyle")

nrows=3
ncols=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4,6), dpi=170)

grid_spec = (512,512)
box_length = (30,30)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

# Get the current working directory
current_directory = os.getcwd()

# Check if the directory matches the pattern "B#.#/S#.#"
if 'B' in current_directory and 'S' in current_directory:
    # Extract the values for B and S
    try:
        b_value = float(current_directory.split('B')[1].split('/')[0])
        s_value = float(current_directory.split('S')[1])
        print(f"B value: {b_value}, S value: {s_value}")
    except (ValueError, IndexError):
        print("Error: Unable to extract B and S values from the directory name.")
else:
    print("The current directory does not match the expected pattern.")
    b_value = -1
    s_value = -1


fig.suptitle('CART-Homopolymer Base Case \n(mRNA, Cat:Lip; 1,1:1) Salt=%.1f, B=%.1f' % (s_value, b_value), fontsize=10)
multi_cam = Camera(fig)


im = []
div = []
cax = [] 
cb = [] 
for i in range(nrows):
    im.append([0] * ncols)
    div.append([0] * ncols)
    cax.append([0] * ncols)
    cb.append([0] * ncols)


axes[0,0].set_title('Composite Image')
axes[0,1].set_title('Max Densities')
axes[1,0].set_title('Cation Channel')
axes[1,1].set_title('mRNA Channel')
axes[2,0].set_title('Lipid Channel')
axes[2,1].set_title('Solvent Channel')

dens_traj = cp.load("live_dens_traj.npy")
print(dens_traj.shape)
S_traj = dens_traj[:,0]
A_traj = dens_traj[:,1]
B_traj = dens_traj[:,2]
L_traj = dens_traj[:,3]
T_traj = dens_traj[:,4]
C_traj = dens_traj[:,5]
charge_max = cp.max(C_traj)

from matplotlib.colors import LinearSegmentedColormap
hold_colors = [(0.9, 0, 0.9, 0), (0.9, 0, 0.9, 1)]
purp_cmap = LinearSegmentedColormap.from_list('purples', hold_colors, N=256)

red_color = [238/256, 102/256, 119/256]
red_cmap = LinearSegmentedColormap.from_list('reds', (red_color + [0], red_color + [1]), N=256)

blue_color = [68/256, 119/256, 170/256]
blue_cmap = LinearSegmentedColormap.from_list('blues', (blue_color + [0], blue_color + [1]), N=256)

cyan_color = [102/256, 204/256, 238/256]
cyan_cmap = LinearSegmentedColormap.from_list('cyans', (cyan_color + [0], cyan_color + [1]), N=256)

green_color = [34/256, 136/256, 51/256]
green_cmap = LinearSegmentedColormap.from_list('greens', (green_color + [0], green_color + [1]), N=256)

yellow_color = [204/256, 187/256, 68/256]
yellow_cmap = LinearSegmentedColormap.from_list('yellows', (yellow_color + [0], yellow_color + [1]), N=256)

grey_color = [187/256, 187/256, 187/256]
grey_cmap = LinearSegmentedColormap.from_list('greys', (grey_color + [0], grey_color + [1]), N=256)

for i in range(0,dens_traj.shape[0]):
    im[0][0] = axes[0,0].imshow(S_traj[i].get(), cmap = blue_cmap, vmin = 0, alpha=0.6)
    im[0][0] = axes[0,0].imshow(B_traj[i].get(), cmap = yellow_cmap, vmin = 0, alpha=.6)
    im[0][0] = axes[0,0].imshow(A_traj[i].get(), cmap = red_cmap, vmin = 0, alpha=.6)
    im[0][0] = axes[0,0].imshow(L_traj[i].get(), cmap = green_cmap, vmin = 0, alpha=.6)
    im[2][1] = axes[2,1].imshow(S_traj[i].get(), cmap = blue_cmap, vmin = 0, alpha=0.6)
    im[1][0] = axes[1,0].imshow(B_traj[i].get(), cmap = yellow_cmap, vmin = 0, alpha=.6)
    im[1][1] = axes[1,1].imshow(A_traj[i].get(), cmap = red_cmap, vmin = 0, alpha=.6)
    im[2][0] = axes[2,0].imshow(L_traj[i].get(), cmap = green_cmap, vmin = 0, alpha=.6)



    S_val = S_traj[i] /100/ cp.average(S_traj[i])
    A_val = A_traj[i] / cp.average(A_traj[i])
    B_val = B_traj[i] / cp.average(B_traj[i])
    L_val = L_traj[i] / cp.average(L_traj[i])

    # Combine the species into a single 4D array
    all_species = cp.stack([S_val, A_val, B_val, L_val], axis=0)

    # Determine the species with the maximum value at each point
    max_species_indices = cp.argmax(all_species, axis=0)

    # Convert CuPy arrays to NumPy arrays
    max_species_indices_np = cp.asnumpy(max_species_indices)

    # Extract unique species indices from the data
    unique_species_indices = np.unique(max_species_indices_np)

    # Create a custom colormap with distinct and pleasing colors for unique species
    colors = list(plt.cm.tab10.colors)
    colors[0] = 'beige'
    colors[1] = 'skyblue'
    colors[2] = 'salmon'
    colors[3] = 'lightseagreen'
    selected_colors = [colors[i] for i in unique_species_indices]
    cmap = mcolors.ListedColormap(selected_colors)

    # Plot the species with the largest value at each point using the custom colormap
    im[0][1] = axes[0,1].imshow(max_species_indices_np, cmap=cmap, origin='lower')


#    im[2][0] = axes[2,0].imshow(T_traj[i].get(), cmap = 'Greys', vmin = 0)
#    fig.tight_layout()
    multi_cam.snap()

where = get_structure_factor(grid, cp.average(L_traj[-10:],axis=0))[0]
L_1D_traj = []
A_1D_traj = []
B_1D_traj = []
S_1D_traj = []
for i in range(1, 11):
    print(i)
    L_1D_traj.append(get_structure_factor(grid, L_traj[-i])[1])
    A_1D_traj.append(get_structure_factor(grid, A_traj[-i])[1])
    B_1D_traj.append(get_structure_factor(grid, B_traj[-i])[1])
    S_1D_traj.append(get_structure_factor(grid, S_traj[-i])[1])
L_1D = cp.average(cp.array(L_1D_traj), axis=0)
A_1D = cp.average(cp.array(A_1D_traj), axis=0)
B_1D = cp.average(cp.array(B_1D_traj), axis=0)
S_1D = cp.average(cp.array(S_1D_traj), axis=0)

reduction = 1.3
struct = plt.figure(figsize=(5.2/reduction,4/reduction))


plt.plot(where.get(), L_1D.get(), label=r'$\textrm{Lipid}$', color=colors[3], linewidth=3)
plt.plot(where.get(), B_1D.get(), label=r'$\textrm{Cation}$', color=colors[2], linewidth=3)
plt.plot(where.get(), A_1D.get(), label=r'$\textrm{mRNA}$', color=colors[1], linewidth=3)
#plt.plot(where.get(), S_1D.get(), label=r'$\textrm{Solvent}$', color='black')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlim(right=30)
plt.xlabel(r"$\textrm{q}$")
plt.ylabel(r"$\textrm{S(q)}$")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.title(r"$\textrm{Structure factor for each component}$")

plt.title(r"$\textrm{$C_{\textrm{s}} = %.1f$, $B = %.1f$}$" % (s_value, b_value,))
if len(sys.argv) > 1:
    tex =  r"$\textrm{(" + sys.argv[1] + r")}$"

    plt.figtext(0.05,0.92, tex, fontsize=20)


left, bottom, width, height = 0.18, 0.23, 0.25, 0.25
ax_imshow = plt.axes([left, bottom, width, height])
ax_imshow.axis('off')



# Display the imshow plot in the subplot
ax_imshow.imshow(max_species_indices_np, cmap=cmap, origin='lower')
ax_imshow.set_title(r"$\textrm{Real space}$", fontsize = 12)


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
        if i==0 and j==1:
            continue
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





#divider = make_axes_locatable(axes[0, 1])
#cax_colorbar = divider.append_axes('right', size='8%', pad=0.02)
#cb_colorbar = fig.colorbar(im[0][1], cax=cax_colorbar, orientation='vertical')
#cb_colorbar.set_label('Species')

#plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()

#exit()





species_labels = ['Solvent', 'mRNA', 'Cation', 'Lipid']


# Create a custom tick locator and formatter for the colorbar
tick_locator = FixedLocator(unique_species_indices)
tick_formatter = FuncFormatter(lambda x, pos: species_labels[int(x)])

cbar = fig.colorbar(im[0][1], cax=cax[0][1], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), cmap=cmap)


cbar.locator = tick_locator
cbar.formatter = tick_formatter
cbar.update_ticks()


fig.tight_layout()
multimation = multi_cam.animate()
multimation.save('movie_traj.gif', writer='pillow')

#struct.savefig(f'structure_factor_S%.1f_B=%.1f.pdf' % (s_value, b_value))
struct.tight_layout()
struct.savefig(f'structure_factor_S{s_value:.1f}_B{b_value:.1f}.pdf')


plt.show()

