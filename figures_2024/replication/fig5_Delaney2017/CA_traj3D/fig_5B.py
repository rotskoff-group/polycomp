import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pymbar import timeseries
from pymbar import testsystems

plt.style.use("stylefile.mplstyle")

only_files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
right_files = [f for f in only_files if f.startswith('traj_')]
right_files.sort()

fig_mu, ax_mu = plt.subplots()
#fig_en, ax_en = plt.subplots()
#fig_para, ax_para = plt.subplots()
points = []
burn_in = 2000
for file in right_files:
    pass
    print(file)
    vol = 625 
    N2 = 40
    N = 80
    D = 3
    T = 10
    A_amt = float(re.findall(r"[-+]?(?:\d*\.*\d+)", file)[0]) 
    traj = np.loadtxt(file, dtype=complex).real
    t0, g, Neff_max = timeseries.detect_equilibration(traj[:,1], nskip=1)
    indices = timeseries.subsample_correlated_data(traj[t0:,1], g=g)
    indices = [i+t0 for i in indices]
#    print(g)
#    print(Neff_max)
#    print(indices)
    points.append([A_amt, np.average(traj[indices,1]), np.var(traj[indices,1])**(0.5), 
        -np.average(traj[indices,2]), -np.var(traj[indices,2])**(0.5),
        np.average(traj[indices,3]), np.var(traj[indices,3])**(0.5), len(indices)])
    #points.append([A_amt, np.average(traj[indices,1]), np.var(traj[indices,1])**(0.5), 
    #    -np.average(traj[indices,2]), -np.var(traj[indices,2])**(0.5),
    #    np.average(traj[indices,3]), np.var(traj[indices,3])**(0.5)])
ref = np.genfromtxt('ref.txt', delimiter=',')
points = np.asarray(points)
#ax_en.plot(points[:,0], points[:,1], color='blue')
#ax_mu.plot(points[:,0], points[:,5], color='blue', label='Calculated $\mu$')
ax_mu.scatter(ref[:,0], ref[:,1], color='black', label=r"$\textrm{Delaney et al.}$",marker = '_')
ax_mu.scatter(points[:,0], points[:,5], color='red', marker = '_', label= r"$\textrm{Numerical $\mu$}$")
ax_mu.errorbar(points[:,0], points[:,5], yerr=points[:,6] / points[:,-1]**(1/2) * 1.69, ls='none', color='red')
#ax_mu.plot(points[:,0], -points[:,3], color='red', label='total pi')
#ax_mu.errorbar(points[:,0], -points[:,3], yerr=points[:,4], ls='none', color='red')
print(points[:,2])
print(points)
#ax_en.errorbar(points[:,0], points[:,1], yerr=points[:,2], ls='none', label= "Numeric Free Energy", color='blue')
#ax_en.legend()
ax_mu.legend()
#ax_en.set_ylabel("Free energy")
ax_mu.set_ylabel(r"$\textrm{$\beta \mu$}$")
#ax_en.set_xlabel("A fraction")
ax_mu.set_xlabel(r"$C$")
ax_mu.set_xlim(10**(-3.5),10**(0.2))
ax_mu.set_ylim(0,6)

#ax_en.set_xscale("log")
ax_mu.set_xscale("log")
#ax_mu.set_yscale("log")
#ax_mu.set_title(r"$\textrm{Comparison to previous work}$")
#ax_para.plot(points[:,3], points[:,5])
#ax_para.set_xlabel("$\mu$")
plt.savefig("fig_5B.pdf",format='pdf')
plt.show()

