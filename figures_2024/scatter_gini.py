#Make the gini plot

import numpy as np
import matplotlib.pyplot as plt
import os

# Apply the style file
stylefile_path = os.path.expanduser('~/polycomp/figures_2024/stylefile.mplstyle')
plt.style.use(stylefile_path)

# Function to calculate Gini coefficient
def gini(arr):
    arr = arr.flatten()
    if np.amin(arr) < 0:
        arr -= np.amin(arr)  # values cannot be negative
    arr += 0.0000001  # values cannot be 0
    arr = np.sort(arr)  # sort
    index = np.arange(1, arr.shape[0] + 1)  # index per array element
    n = arr.shape[0]
    return ((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))  # Gini coefficient

# Load the array
array = np.load('../process/merged_array.npy')                                           

# Extract the last value from each row (y-axis data)                                     
last_values = array[:, 7]                                                                
                                                                                          
# Calculate the Gini coefficient for the values from the second to the seventh column (x-axis data)
gini_coefficients = np.array([gini(row[1:7]) for row in array])

# Calculate the line of best fit                                                         
coefficients = np.polyfit(gini_coefficients, last_values, 1)                             
slope, intercept = coefficients
polynomial = np.poly1d(coefficients)                                                     
best_fit_line = polynomial(gini_coefficients)                                            

# Calculate R² value
residuals = last_values - best_fit_line                                                  
ss_res = np.sum(residuals**2)                                                            
ss_tot = np.sum((last_values - np.mean(last_values))**2)                                 
r_squared = 1 - (ss_res / ss_tot) 

# Create the scatter plot
line_label = rf"$y = {slope:.3f}x {intercept:+.3f},$" + " " + rf"$R^2 = {r_squared:.3f}$" 
plt.figure(figsize=(10, 6))                                                              
plt.scatter(gini_coefficients, last_values, alpha=0.2)                                   
plt.plot(gini_coefficients, best_fit_line, color='red', label=line_label, lw=3)
plt.xlabel(r"$\textrm{Gini Coefficient of Species Fractions}$", fontsize=24) 
plt.ylabel(r"$\textrm{Average Error}$", fontsize=24)
plt.legend(fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust the size as needed


# Adjusting the position of the R², slope, and intercept annotation                      
#plt.text(
#    0.05, 0.85,  # Adjust the position as needed
#    f'$R^2 = {r_squared:.3f}$\n'
#    f'Slope = {slope:.3f}\n'
#    f'Intercept = {intercept:.3f}',
#    transform=plt.gca().transAxes,
#    fontsize=12, verticalalignment='top',
#    bbox=dict(facecolor='white', alpha=0.5)  # Background box for better readability
#)
#plt.text(
#    0.05, 0.85,  # Adjust the position as needed
#    rf"$R^2 = {r_squared:.3f}$" + "\n" +
#    rf"$\textrm{{Slope}} = {slope:.3f}$" + "\n" +
#    rf"$\textrm{{Intercept}} = {intercept:.3f}$",
#    transform=plt.gca().transAxes,
#    fontsize=22, verticalalignment='top',
#    bbox=dict(facecolor='white', alpha=0.5)  # Background box for better readability
#)


# Adjust plot margins to avoid clipping
plt.subplots_adjust(top=0.85)

# Remove grid lines
plt.grid(False)

plt.savefig('gini.pdf')
#plt.show()

