import os
import sys
sys.path.insert(0, "/scratch/users/epert/nice_plotting/salt_tests/nanobrush")
import cupy as cp
import numpy as np
import math
#import matplotlib
#matplotlib.use('AGG')  #or PDF, SVG, or PS

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("stylefile.mplstyle")

nrows=1
ncols=1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8/3,9.4/3), dpi=1000)
fig.suptitle(r"$\textrm{$B=2.5$}, \textrm{$C_{\textrm{s}}=0.0$}$", fontsize=16, y=0.10)

im = []
div = []
cax = [] 
cb = [] 
for i in range(nrows):
    im.append([0] * ncols)
    div.append([0] * ncols)
    cax.append([0] * ncols)
    cb.append([0] * ncols)



dens_traj = cp.load("raw_data/dens_B2.5S0_last_frame.npy")
print(dens_traj.shape)
S_traj = dens_traj[0]
A_traj = dens_traj[1]
B_traj = dens_traj[2]
L_traj = dens_traj[3]
T_traj = dens_traj[4]
C_traj = dens_traj[5]
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

# Create a custom colormap with distinct and pleasing colors for unique species
colors = list(plt.cm.tab10.colors)
colors[0] = blue_color
colors[1] = red_color
colors[2] = yellow_color
colors[3] = green_color

colors[0] = 'beige'
colors[1] = 'skyblue'
colors[2] = 'salmon'
colors[3] = 'lightseagreen'

dens_traj = cp.load("raw_data/dens_B2.5S0_last_frame.npy")
dens_traj = np.flip(np.roll(dens_traj, (150,0), axis=(1,2)), axis=1)
S_traj = dens_traj[0]
A_traj = dens_traj[1]
B_traj = dens_traj[2]
L_traj = dens_traj[3]
T_traj = dens_traj[4]
C_traj = dens_traj[5]
S_val = S_traj / cp.average(S_traj) /2.2
A_val = A_traj / cp.average(A_traj)
B_val = B_traj / cp.average(B_traj)
L_val = L_traj / cp.average(L_traj)

# Combine the species into a single 4D array
all_species = cp.stack([S_val, A_val, B_val, L_val], axis=0)

# Determine the species with the maximum value at each point
max_species_indices = cp.argmax(all_species, axis=0)

# Convert CuPy arrays to NumPy arrays
max_species_indices_np = cp.asnumpy(max_species_indices)

# Extract unique species indices from the data
unique_species_indices = np.unique(max_species_indices_np)


selected_colors = [colors[l] for l in unique_species_indices]
cmap = mcolors.ListedColormap(selected_colors)

# Plot the species with the largest value at each point using the custom colormap
im[0][0] = axes.imshow(max_species_indices_np, cmap=cmap, origin='lower')


species_labels = ['Solvent', 'mRNA', 'Cation', 'Lipid']


# Create a custom tick locator and formatter for the colorbar
tick_locator = FixedLocator(unique_species_indices)
tick_formatter = FuncFormatter(lambda x, pos: species_labels[int(x)])

#    im[2][0] = axes[2,0].imshow(T_traj[i].get(), cmap = 'Greys', vmin = 0)
#    fig.tight_layout()


#divider = make_axes_locatable(axes[0, 1])
#cax_colorbar = divider.append_axes('right', size='8%', pad=0.02)
#cb_colorbar = fig.colorbar(im[0][1], cax=cax_colorbar, orientation='vertical')
#cb_colorbar.set_label('Species')

#plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()

#exit()
axes.set_xticks([])
axes.set_yticks([])


aspect = 1
#cbar_1 = fig.colorbar(im[0][0], cax=cax[0][0], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), aspect=aspect)
#cbar_2 = fig.colorbar(im[0][1], cax=cax[0][1], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), aspect=aspect)
#cbar_3 = fig.colorbar(im[0][2], cax=cax[0][2], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), aspect=aspect)

#for cbar in [cbar_1, cbar_2, cbar_3]:
#    cbar.locator = tick_locator
#    cbar.formatter = tick_formatter
#    cbar.update_ticks()
#    cbar.set_ticklabels(species_labels, fontsize=8)







fig.subplots_adjust(top=1., bottom=0.15)
fig.tight_layout(rect=[0, 0.15, 1, 1.])


plt.savefig("dens_B2.5C0.0.pdf", dpi=300)
