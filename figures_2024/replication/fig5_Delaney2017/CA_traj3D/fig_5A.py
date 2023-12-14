import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    z = np.array(z, subok=True, copy=False, ndmin=1)

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

N = 10
np.random.seed(101)
x = np.random.rand(N)
y = np.random.rand(N)
fig, ax = plt.subplots()


#My code
only_files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
right_files = [f for f in only_files if f.startswith('traj_')]
right_files.sort()

values = []
for file in right_files:
    pass
    print(file)
    A_amt = float(re.findall(r"[-+]?(?:\d*\.*\d+)", file)[0]) 
    traj = np.loadtxt(file, dtype=complex).real
    values.append([A_amt, np.average(traj[-1000:,1]), np.var(traj[-1000:,1])**(0.5), 
        np.average(traj[2000:,3]), np.var(traj[2000:,3])**(0.5), 
        np.average(traj[2000:,2]), np.var(traj[2000:,2])**(0.5)])
values = np.asarray(values)
print(values)
interpolation = 20
path = mpath.Path(np.column_stack([values[:,3], values[:,5]]))
zpath = mpath.Path(np.column_stack([values[:,3], values[:,0]]))
verts = path.interpolated(steps=interpolation).vertices
zverts = zpath.interpolated(steps=interpolation).vertices
x, y, z = verts[:, 0], verts[:, 1], zverts[:, 1]
#z = np.interp(values[:,0]
#print(z.shape)
#print(x.shape)
#exit()
norm = mpl.colors.LogNorm(vmin=np.amin(z), vmax=np.amax(z))
div = make_axes_locatable(ax)
cmap = mpl.cm.cividis
colorline(x, y, z, cmap=cmap, linewidth=2, norm=norm)
plt.errorbar(values[:,3], values[:,5], xerr=values[:,4] * 0, yerr=values[:,6] * 0, ls='none', color="red", elinewidth=1)


cax = div.append_axes('right', size='4%', pad=0.02)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical', label='Fraction A')
ax.set_xlabel('$\mu_A$')
ax.set_ylabel('$\Pi$')
plt.show()


