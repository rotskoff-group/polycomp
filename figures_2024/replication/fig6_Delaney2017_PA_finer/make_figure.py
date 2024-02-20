import matplotlib.pyplot as plt
import numpy as np
import os
import re
#import sys

#sys.path.insert(0, "/polycomp_main/figures_2024")
plt.style.use("stylefile.mplstyle")

#get all directories that start with 'gibbs'
dirs = [d for d in os.listdir('.') if re.match(r'gibbs_E', d)]
#dirs = [d for d in os.listdir('..')]

dirs = ['./' + d for d in dirs]
dirs.sort()

print(dirs)
#get the number at the end of each directory
nums = [int(d.split('_E')[-1]) for d in dirs]
print(nums)

#sort the directories
dirs.sort()

#get the file 'gibbs_traj.npy' from each directory
trajs = [np.loadtxt(d+'/traj.npy', dtype=complex) for d in dirs]

#get the average value of the last 400 points in each trajectory, using only columns 2 and 3
means = [np.mean(traj[-100:,0:2], axis=0) * 2 for traj in trajs]


#get the corresponding variances
variances = [np.var(traj[-100:,0:2], axis=0) for traj in trajs]


ref = np.genfromtxt('ref.txt', delimiter=',')

#plot the number versus means of the original directory
fig, ax = plt.subplots()
ax.set_title(r"$\textrm{Coacervate Phase for Polyampholyte at $B=0.5$ and $C_s=0.0$}$", fontsize=15)
print([m[0].real for m in means])
print(nums)
ax.scatter([m[0].real for m in means], nums, label='Supernatant Concentration', color='blue')
ax.scatter([m[1].real for m in means], nums, label='Coacervate Concentration', color='red')
#ax.errorbar([m[0] for m in means], nums, yerr=[v[0] for v in variances], fmt='o', label='x', color='blue')
#ax.errorbar([m[1] for m in means], nums, yerr=[v[1] for v in variances], fmt='o', label='y', color='blue')

ax.scatter(ref[:,0], ref[:,1], color='black', label='Reference Result', marker='x')

#set x axis log
ax.set_xscale('log')
ax.legend()
ax.set_ylabel(r"$\textrm{E}$", fontsize=12)
ax.set_xlabel(r"$\textrm{C (reduced concentration)}$", fontsize=12)

#fig.suptitle(r"$\textrm{Phase coexistence for polyampholyte in implicit solvent}$")


fig.savefig('Fig6_rep.pdf', format='pdf')
plt.show()

