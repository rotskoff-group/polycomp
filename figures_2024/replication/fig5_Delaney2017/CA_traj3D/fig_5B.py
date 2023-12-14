import matplotlib.pyplot as plt
import numpy as np
import os
import re

only_files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
right_files = [f for f in only_files if f.startswith('traj_')]
right_files.sort()

fig_mu, ax_mu = plt.subplots()
fig_en, ax_en = plt.subplots()
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
    points.append([A_amt, np.average(traj[burn_in:,1]), np.var(traj[burn_in:,1])**(0.5), 
        -np.average(traj[burn_in:,2]), -np.var(traj[burn_in:,2])**(0.5),
        np.average(traj[burn_in:,3]), np.var(traj[burn_in:,3])**(0.5)])
    #points.append([A_amt, np.average(traj[burn_in:,1]), np.var(traj[burn_in:,1])**(0.5), 
    #    np.average(traj[burn_in:,2]), np.var(traj[burn_in:,2])**(0.5) / float(len(traj[burn_in:,2])/20)**(0.5), 
    #    np.average(traj[burn_in:,3]), np.var(traj[burn_in:,3])**(0.5) / float(len(traj[burn_in:,3])/20)**(0.5)])

#for A_amt in  np.linspace(0.01,.99, 100):
#    points.append([A_amt, vol * 3 * A_amt * (1 - A_amt) + 
#        vol * A_amt * (np.log(A_amt)) + 
#        vol * (1-A_amt) * (np.log((1 - A_amt))), np.var(traj[-1000:])])
#points1 = np.asarray(points1)
ref = np.genfromtxt('ref.txt', delimiter=',')
points = np.asarray(points)
ax_en.plot(points[:,0], points[:,1], color='blue')
#ax_mu.plot(points[:,0], points[:,3], color='orange')
ax_mu.plot(points[:,0], points[:,5], color='blue', label='chemical potential')
#ax_mu.errorbar(points[:,0], points[:,5], yerr=points[:,6], ls='none', color='blue')
ax_mu.plot(points[:,0], -points[:,3], color='red', label='total pi')
#ax_mu.errorbar(points[:,0], -points[:,3], yerr=points[:,4], ls='none', color='red')
#ax_mu.plot(ref[:,0], ref[:,1], color='black')
ax_mu.scatter(ref[:,0], ref[:,1], color='black', label="Fredrickson's $\mu$")
print(points[:,2])
print(points)
ax_en.errorbar(points[:,0], points[:,1], yerr=points[:,2], ls='none', label= "Numeric Free Energy", color='blue')
#ax_mu.errorbar(points[:,0], points[:,3], yerr=points[:,4], ls='none', label= "$\mu$", color='orange')
#ax_mu.errorbar(points[:,0], points[:,5], yerr=points[:,6], ls='none', label= "$\Pi$", color='red')
#ax_en.errorbar(points[:,0], points[:,1], yerr=points[:,2], ls='none')
#ax_en.errorbar(points[:,0], points[:,1], yerr=points[:,2], ls='none')
#ax_en.scatter(points1[:,0], points1[:,1] + points2[:,1])
ax_en.legend()
ax_mu.legend()
ax_en.set_ylabel("Free energy")
ax_mu.set_ylabel("chemical potential")
ax_en.set_xlabel("A fraction")
ax_mu.set_xlabel("A fraction")

ax_en.set_xscale("log")
ax_mu.set_xscale("log")
#ax_mu.set_yscale("log")
fig_mu.suptitle("Chemical potential and Pressure Replication at E=400")
#ax_para.plot(points[:,3], points[:,5])
#ax_para.set_xlabel("$\mu$")
plt.show()
