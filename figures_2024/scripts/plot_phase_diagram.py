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

nrows=2
ncols=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8/1.1,20/3.3), dpi=1000)

im = []
div = []
cax = [] 
cb = [] 
for i in range(nrows):
    im.append([0] * ncols)
    div.append([0] * ncols)
    cax.append([0] * ncols)
    cb.append([0] * ncols)



dens_traj = cp.load("raw_data/B0.5S0_last_frame.npy")
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

#Start of actual plot
for i in range(ncols):
    for j in range(nrows):
        if i==0 and j==0:
            continue
        if i==1 and j==0:
            dens_traj = cp.load("raw_data/B0.5S0_last_frame.npy")
            dens_traj = np.flip(np.roll(dens_traj, (150,0), axis=(1,2)), axis=1)
            axes[i][j].set_title(r"$\textrm{$B=0.5$ and $C_{\textrm{s}}=0.0$}$", fontsize=12)
        if i==1 and j==1:
            dens_traj = cp.load("raw_data/B4S0_last_frame.npy")
            dens_traj = np.flip(np.roll(dens_traj, 240, axis=2), axis=1)
            axes[i][j].set_title(r"$\textrm{$B=4.0$ and $C_{\textrm{s}}=0.0$}$", fontsize=12)
        if i==0 and j==1:
            dens_traj = cp.load("raw_data/B5S3_last_frame.npy")
            dens_traj = np.flip(np.roll(dens_traj, 130, axis=2), axis=1)
            axes[i][j].set_title(r"$\textrm{$B=5.0$ and $C_{\textrm{s}}=3.0$}$", fontsize=12)
        
#        im[i][j] = axes[i,j].imshow(np.flip(np.roll(max_species_indices_np, 240, axis=1), axis=0), cmap=cmap, origin='lower')
        S_traj = dens_traj[0]
        A_traj = dens_traj[1]
        B_traj = dens_traj[2]
        L_traj = dens_traj[3]
        T_traj = dens_traj[4]
        C_traj = dens_traj[5]
        S_val = S_traj / cp.average(S_traj)
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
        im[i][j] = axes[i,j].imshow(max_species_indices_np, cmap=cmap, origin='lower')


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


#Random plotting garbage, sure there's a better way to do this
first = True
for part in axes:
    for ax in part:
        if first:
            first = False
            continue
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
        cax[i][j] = div[i][j].append_axes('right', size='1%', pad=0.02)
for i in range(nrows):
    for j in range(ncols):
#        if i==0 and j==0:
#            continue
        if im[i][j]==0:
            continue
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')
        if nrows==1:
            div[i][j] = make_axes_locatable(axes[j])
        else:
            div[i][j] = make_axes_locatable(axes[i,j])
        cax[i][j] = div[i][j].append_axes('right', size='4%', pad=0.02)
        cb[i][j].remove()
        cb[i][j] = fig.colorbar(im[i][j], cax=cax[i][j], orientation='vertical')

aspect = 1
cbar_1 = fig.colorbar(im[0][1], cax=cax[0][1], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), aspect=aspect)
cbar_2 = fig.colorbar(im[1][0], cax=cax[1][0], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), aspect=aspect)
cbar_3 = fig.colorbar(im[1][1], cax=cax[1][1], ticks=unique_species_indices, boundaries=np.linspace(unique_species_indices.min()-0.5, unique_species_indices.max()+0.5, len(unique_species_indices)+1), aspect=aspect)

for cbar in [cbar_1, cbar_2, cbar_3]:
    cbar.locator = tick_locator
    cbar.formatter = tick_formatter
    cbar.update_ticks()
    cbar.set_ticklabels(species_labels, fontsize=8)







# Directory where your data is stored
data_directory = "raw_data/"

b_values = np.arange(0.0, 6.1, 0.5)
s_values = np.arange(0.0, 5.1, 0.2)
b_values, s_values = np.meshgrid(b_values, s_values, indexing="ij")
b_values = b_values.flatten()
s_values = s_values.flatten()

processed_L_order_values = np.load(data_directory + "L_order.npy")
processed_C_order_values = np.load(data_directory + "C_order.npy")

# Create a 2D array corresponding to B and S values
unique_b_values = np.unique(b_values)
unique_s_values = np.unique(s_values)
processed_L_order_grid = np.zeros((len(unique_s_values), len(unique_b_values)))
processed_C_order_grid = np.zeros((len(unique_s_values), len(unique_b_values)))

# Populate the processed grids with the corresponding values
for b, s, L_order_value, C_order_value in zip(
    b_values, s_values, processed_L_order_values, processed_C_order_values
):
    row_idx = np.where(unique_s_values == s)[0][0]
    col_idx = np.where(unique_b_values == b)[0][0]
    processed_L_order_grid[row_idx, col_idx] = L_order_value
    processed_C_order_grid[row_idx, col_idx] = C_order_value


phase = np.zeros_like(processed_L_order_grid)
where_1 = np.where((processed_L_order_grid < 2.4))
phase[where_1] = 1
where_2 = np.where((processed_C_order_grid < 1.001))
phase[where_2] = 2
where_3 = np.where(
    np.logical_and(
        processed_L_order_grid < 1.15,
        np.logical_and(processed_C_order_grid >= 1, processed_C_order_grid < 1.0016),
    )
)
phase[where_3] = 3
plt.subplot(221)
plt.title(r"$\textrm{Phase Diagram}$")
## Define your custom colors for the 4 values
# colors = ['salmon', 'skyblue', 'wheat', 'grey']

# Create a ListedColormap with the custom colors
# from matplotlib.colors import ListedColormap
# cmap = ListedColormap(colors)

# phase_plot = plt.imshow(phase, cmap=cmap, origin='lower', extent=[np.min(unique_b_values), np.max(unique_b_values), np.min(unique_s_values), np.max(unique_s_values)], aspect='auto')


# cbar = plt.colorbar(phase_plot, ticks=[3/8.0, 9/8.0, 15/8.0, 21/8.0])
# cbar.set_ticklabels(['Lamellar', 'Coacervate-Core', 'Lipid-Core', 'Homogeneous'], rotation=90, ha='center', va='center')
from matplotlib.colors import ListedColormap

colors = ["salmon", "skyblue", "wheat", "gainsboro"]
cmap = ListedColormap(colors)

markers = ['*', 'o', 's', '^']

# Create meshgrid for scatter plot


b, s = np.meshgrid(unique_b_values, unique_s_values)

scatter_plot = plt.scatter(
    b, s, c=phase.flatten(), cmap=cmap, edgecolors="black", linewidth=0.5, s=0.0001
)

for i in range(4):
    where = np.where(phase == i)
    plt.scatter(b[where], s[where], c=colors[i], marker=markers[i], s=20)
cbar = plt.colorbar(scatter_plot, ticks=[3 / 8.0, 9 / 8.0, 15 / 8.0, 21 / 8.0], aspect = 25)
cbar.set_ticklabels(
    ["Lamellar", "Coacervate", "Lipid-Core", "Homogeneous"],
    rotation=0,
    ha="left",
    va="center",
    fontsize=8
)

axes[0][0].tick_params(axis='both', labelsize=15)
plt.axis("auto")
plt.xlabel("$B$")
plt.ylabel(r"$C_{\textrm{s}}$")
for i, ax in enumerate(axes.flat):
    label = r"$\textrm{(" + chr(97 + i) + r")}$"
    ax.text(-0.15, 1.15, label, transform=ax.transAxes, va='top', ha='left', fontsize=17)

fig.tight_layout()
plt.savefig("phase_diagram.pdf", dpi=300)














#Start the second plot
# Plot for the L_order value


colors = [(0.9803921568627451, 0.5019607843137255, 0.4470588235294118, 0), (0.9803921568627451, 0.5019607843137255, 0.4470588235294118, 1)]  # White to Salmon

# Create the colormap
cmap_name = 'white_to_salmon'
cmap_salmon = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

colors = [(0.5294117647058824, 0.807843137254902, 0.9215686274509803, 0), (0.5294117647058824, 0.807843137254902, 0.9215686274509803, 1)]  # White to Skyblue

# Create the colormap
cmap_name = 'white_to_lightseagreen'
cmap_skyblue = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

colors = [(0.5294117647058824, 0.807843137254902, 0.9215686274509803, 0), (0.5294117647058824, 0.807843137254902, 0.9215686274509803, 0.5)]  # White to Skyblue (half opacity)
cmap_sb_transp = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
const2 = 18
green_col = [
    (0.9686274509803922, 0.9882352941176471, 0.9607843137254902, 1.0/const2),
    (0.8980392156862745, 0.9607843137254902, 0.8784313725490196, 2.0/const2),
    (0.7803921568627451, 0.9137254901960784, 0.7529411764705882, 3.0/const2),
    (0.6313725490196078, 0.8509803921568627, 0.6078431372549019, 4.0/const2),
    (0.4549019607843137, 0.7686274509803922, 0.4627450980392157, 5.0/const2),
    (0.2549019607843137, 0.6705882352941176, 0.36470588235294116, 6.0/const2),
    (0.13725490196078433, 0.5450980392156862, 0.27058823529411763, 7.0/const2),
    (0.0, 0.42745098039215684, 0.17254901960784313, 8.0/const2),
    (0.0, 0.26666666666666666, 0.10588235294117647, 9.0/const2)
]
green_col = [
    (0.0, 0.26666666666666666, 0.10588235294117647, 0.0/const2),
    (0.0, 0.26666666666666666, 0.10588235294117647, 9.0/const2)
]

green_transp = LinearSegmentedColormap.from_list(cmap_name, green_col, N=256) 


fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(14.0/1.25, 5.5/1.25),dpi=500)
for ax in axes2:
    ax.tick_params(axis='both', labelsize=15)

plt.subplot(131)
plt.imshow(
    processed_L_order_grid,
    cmap=cmap_skyblue,
    origin="lower",
    extent=[
        np.min(unique_b_values),
        np.max(unique_b_values),
        np.min(unique_s_values),
        np.max(unique_s_values),
    ],
    aspect=1.2,
)
# plt.colorbar(label='Processed L_order Value')
plt.xlabel("$B$")
plt.ylabel(r"$C_{\textrm{s}}$")
plt.title(r"$\textrm{Lipid Order Parameter}$")

# Plot for the C_order value
const = 9
red_col = [
    (1.0, 0.9607843137254902, 0.9411764705882353, 0.0),
    (1.0, 0.9607843137254902, 0.9411764705882353, 1.0/const),
    (0.996078431372549, 0.8784313725490196, 0.8235294117647058, 2.0/const),
    (0.9882352941176471, 0.7333333333333333, 0.6313725490196078, 3.0/const),
    (0.9882352941176471, 0.5725490196078431, 0.4470588235294118, 4.0/const),
    (0.984313725490196, 0.41568627450980394, 0.2901960784313726, 5.0/const),
    (0.9372549019607843, 0.23137254901960785, 0.17254901960784313, 6.0/const),
    (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 7.0/const),
    (0.6470588235294118, 0.058823529411764705, 0.08235294117647059, 8.0/const),
    (0.403921568627451, 0.0, 0.050980392156862744, 9.0/const)
]

red_transp = LinearSegmentedColormap.from_list(cmap_name, red_col, N=256) 
plt.subplot(132)
print(np.amin(processed_C_order_grid))
plt.imshow(
    processed_C_order_grid,
    cmap=cmap_salmon,
    origin="lower",
    extent=[
        np.min(unique_b_values),
        np.max(unique_b_values),
        np.min(unique_s_values),
        np.max(unique_s_values),
    ],
    aspect=1.2,
)
# plt.colorbar(label='Processed C_order Value')
plt.xlabel("$B$")
plt.ylabel(r"$C_{\textrm{s}}$")
plt.title(r"$\textrm{Ion Order Parameter}$")

# Overlay plot for the L_order and C_order values
plt.subplot(133)
plt.imshow(
    processed_C_order_grid,
    cmap=cmap_salmon,
    origin="lower",
    extent=[
        np.min(unique_b_values),
        np.max(unique_b_values),
        np.min(unique_s_values),
        np.max(unique_s_values),
    ],
    aspect=1.2,
)
plt.imshow(
    processed_L_order_grid,
    cmap=cmap_sb_transp,
    origin="lower",
    extent=[
        np.min(unique_b_values),
        np.max(unique_b_values),
        np.min(unique_s_values),
        np.max(unique_s_values),
    ],
    aspect=1.2,
)
# plt.colorbar(label='Processed Values')
plt.xlabel("$B$")
plt.ylabel(r"$C_{\textrm{s}}$")
plt.title(r"$\textrm{Overlay of Both Parameters}$")

for i, ax in enumerate(axes2.flat):
    label = r"$\textrm{(" + chr(97 + i) + r")}$"
    ax.text(-0.2, 1.15, label, transform=ax.transAxes, va='top', ha='left', fontsize=17)
fig.tight_layout()
plt.savefig("order_parameters.pdf", dpi=300)

#plt.show()

