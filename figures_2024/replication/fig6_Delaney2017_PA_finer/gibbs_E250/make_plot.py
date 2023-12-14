import matplotlib.pyplot as plt
import numpy as np
import os
import re

#only_files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
#right_files = [f for f in only_files if f.startswith('traj_')]
#right_files.sort()

traj = np.loadtxt('traj.npy', dtype=np.complex_)
vol_traj = np.loadtxt('vol_traj.npy', dtype=np.complex_)
fig, ax = plt.subplots(2,2)
spec = (traj.shape[1]) // 4
print(spec)
print(traj.shape)
ax[0,0].plot(traj[:,2 * spec])
ax[0,0].plot(traj[:,3 * spec]) 
ax[1,0].plot(traj[:,0])
ax[1,0].plot(traj[:,spec]) 

ax[0,1].plot(vol_traj[:,2])
ax[0,1].plot(vol_traj[:,3])
ax[1,0].set_yscale('log')
ax[1,1].plot(vol_traj[:,0])
ax[1,1].plot(vol_traj[:,1])

ax[0,0].set_title('$\Delta \mu$')
ax[0,0].set_xlabel('Gibbs Steps')
ax[0,0].set_ylabel('$\mu$')

ax[0,1].set_title('$\Delta \Pi$')
ax[0,1].set_xlabel('Gibbs Steps')
ax[0,1].set_ylabel('$\Pi$')

ax[1,0].set_title('Concentrations')
ax[1,0].set_xlabel('Gibbs Steps')
ax[1,0].set_ylabel('C')

ax[1,1].set_title('Volumes')
ax[1,1].set_xlabel('Gibbs Steps')
ax[1,1].set_ylabel('V')


fig.tight_layout()

plt.show()
exit()

fig_mu, ax_mu = plt.subplots()
fig_en, ax_en = plt.subplots()
#fig_para, ax_para = plt.subplots()
points = []
samples = 3500
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
    #points.append([A_amt, np.average(traj[-1000:]) + 
    #    vol * A_amt * (np.log(A_amt) - 1) + 
    #    vol * (1-A_amt) * (np.log((1 - A_amt)) - 1), np.var(traj[-1000:])])
    #points1.append([A_amt, 
    #    vol * A_amt * (np.log(A_amt) - 1) + 
    #    vol * (1-A_amt) * (np.log((1 - A_amt)) - 1)])
    points.append([A_amt, np.average(traj[-samples:,1]), np.var(traj[-samples:,1])**(0.5), 
        np.average(traj[-samples:,2]), np.var(traj[-samples:,2])**(0.5), 
        np.average(traj[-samples:,3]), np.var(traj[-samples:,3])**(0.5)])

#for A_amt in  np.linspace(0.01,.99, 100):
#    points.append([A_amt, vol * 3 * A_amt * (1 - A_amt) + 
#        vol * A_amt * (np.log(A_amt)) + 
#        vol * (1-A_amt) * (np.log((1 - A_amt))), np.var(traj[-1000:])])
#points1 = np.asarray(points1)
points = np.asarray(points)
ax_en.plot(points[:,0], points[:,1], color='blue')
ax_mu.plot(points[:,0], points[:,3], color='orange')
ax_mu.plot(points[:,0], points[:,5], color='red')
print(points[:,2])
print(points)
ax_en.errorbar(points[:,0], points[:,1], yerr=points[:,2], ls='none', label= "Numeric Free Energy", color='blue')
ax_mu.errorbar(points[:,0], points[:,3], yerr=points[:,4], ls='none', label= "$\mu$", color='orange')
ax_mu.errorbar(points[:,0], points[:,5], yerr=points[:,6], ls='none', label= "$\Pi$", color='red')
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
#fig_en.suptitle("AB homopolymers, $\chi=3$")
#ax_para.plot(points[:,3], points[:,5])
#ax_para.set_xlabel("$\mu$")
plt.show()
