import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where your data is stored
data_directory = "raw_data/"

b_values = np.arange(0.0,6.1,0.5)
s_values = np.arange(0.0,5.1,0.2)
b_values, s_values = np.meshgrid(b_values, s_values, indexing='ij')
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
for b, s, L_order_value, C_order_value in zip(b_values, s_values, processed_L_order_values, processed_C_order_values):
    row_idx = np.where(unique_s_values == s)[0][0]
    col_idx = np.where(unique_b_values == b)[0][0]
    processed_L_order_grid[row_idx, col_idx] = L_order_value
    processed_C_order_grid[row_idx, col_idx] = C_order_value

# Plot for the L_order value
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(processed_L_order_grid, cmap='Reds', origin='lower', extent=[np.min(unique_b_values), np.max(unique_b_values), np.min(unique_s_values), np.max(unique_s_values)], aspect='auto')
#plt.colorbar(label='Processed L_order Value')
plt.xlabel('B Values')
plt.ylabel('S Values')
plt.title('Lipid Order Parameter')

# Plot for the C_order value
plt.subplot(222)
print(np.amin(processed_C_order_grid))
plt.imshow(processed_C_order_grid, cmap='Greens', origin='lower', extent=[np.min(unique_b_values), np.max(unique_b_values), np.min(unique_s_values), np.max(unique_s_values)], aspect='auto')
#plt.colorbar(label='Processed C_order Value')
plt.xlabel('B Values')
plt.ylabel('S Values')
plt.title('Ion Order Parameter')

# Overlay plot for the L_order and C_order values
plt.subplot(223)
plt.imshow(processed_L_order_grid, cmap='Reds', origin='lower', extent=[np.min(unique_b_values), np.max(unique_b_values), np.min(unique_s_values), np.max(unique_s_values)], aspect='auto')
plt.imshow(processed_C_order_grid, cmap='Greens', alpha=0.5, origin='lower', extent=[np.min(unique_b_values), np.max(unique_b_values), np.min(unique_s_values), np.max(unique_s_values)], aspect='auto')
#plt.colorbar(label='Processed Values')
plt.xlabel('B Values')
plt.ylabel('S Values')
plt.title('Overlay of Both Parameters')

phase = np.zeros_like(processed_L_order_grid)
where_1 = np.where((processed_L_order_grid < 2.4))
phase[where_1] = 1
where_2 = np.where((processed_C_order_grid < 1.001))
phase[where_2] = 2
where_3 = np.where(np.logical_and(processed_L_order_grid < 1.15, np.logical_and(processed_C_order_grid >= 1, processed_C_order_grid < 1.0016)))
phase[where_3] = 3 
plt.subplot(224)
## Define your custom colors for the 4 values
#colors = ['salmon', 'skyblue', 'wheat', 'grey']

# Create a ListedColormap with the custom colors
#from matplotlib.colors import ListedColormap
#cmap = ListedColormap(colors)

#phase_plot = plt.imshow(phase, cmap=cmap, origin='lower', extent=[np.min(unique_b_values), np.max(unique_b_values), np.min(unique_s_values), np.max(unique_s_values)], aspect='auto')


#cbar = plt.colorbar(phase_plot, ticks=[3/8.0, 9/8.0, 15/8.0, 21/8.0])
#cbar.set_ticklabels(['Lamellar', 'Coacervate-Core', 'Lipid-Core', 'Homogeneous'], rotation=90, ha='center', va='center')
from matplotlib.colors import ListedColormap
colors = ['salmon', 'skyblue', 'wheat', 'grey']
cmap = ListedColormap(colors)

# Create meshgrid for scatter plot


b, s = np.meshgrid(unique_b_values, unique_s_values)

scatter_plot = plt.scatter(b, s, c=phase.flatten(), cmap=cmap, edgecolors='black', linewidth=0.5, s=200)
cbar = plt.colorbar(scatter_plot, ticks=[3/8.0, 9/8.0, 15/8.0, 21/8.0])
cbar.set_ticklabels(['Lamellar', 'Coacervate-Core', 'Lipid-Core', 'Homogeneous'], rotation=90, ha='center', va='center')
plt.axis('auto')

plt.xlabel('B values')
plt.ylabel('S values')
plt.title('Scatter Plot with Color Coded Phases')

plt.tight_layout()

plt.show()

exit()
