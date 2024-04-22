import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

agent1 = True
agent2 = False


def pointInRect(points):
    for p in points:
        if np.sqrt((p[2] - p[0]) ** 2 + ((70 - p[3]) - p[1]) ** 2) <= 1.5:
            return True
    return False

#________________________________________________________________________________________

# file = 'closed_loop/sine/closedloop_traj_pinn_narrowroad_sine.mat'
# file = 'closed_loop/sine/closedloop_traj_valuehardening_narrowroad_sine.mat'
# file = 'closed_loop/sine/closedloop_traj_supervised_narrowroad_sine.mat'
file = 'closed_loop/sine/closedloop_traj_hybrid_narrowroad_sine.mat'
# file = 'closed_loop/sine/closedloop_traj_epigraphical_narrowroad_sine.mat'
index = 2
# title = 'PINN'
# title = 'Value hardening'
# title = 'Supervised'
title = 'Hybrid'
# title = 'Epigraphical'
special = 0

#____________________________________________________________________________________________________

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)

data = scipy.io.loadmat(file)
X = data['X']
V = data['V']
T = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
X0 = X[:, idx0]
count = 0
traj_list = []

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

if agent1 == True:
    # norm = plt.Normalize(np.min(V[0, :]), 0)
    norm = plt.Normalize(-3000, 0)
if agent2 == True:
    norm = plt.Normalize(np.min(V[1, :]), 0)

for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        dx1 = X[1, idx0[n - 1]:]
        dy1 = X[0, idx0[n - 1]:]
        dx2 = X[5, idx0[n - 1]:]
        dy2 = X[4, idx0[n - 1]:]

        V1 = V[0, idx0[n - 1]:]
        V2 = V[1, idx0[n - 1]:]
        t = T[0, idx0[n - 1]:]

        dist = np.sqrt((dx2 - dx1) ** 2 + ((70 - dy2) - dy1) ** 2)

        pairs = zip(dx1, dy1, dx2, dy2)
        if pointInRect(pairs):
            count += 1
            print(n - 1)
            traj_list.append(n - 1)
    else:
        dx1 = X[1, idx0[n - 1]: idx0[n]]
        dy1 = X[0, idx0[n - 1]: idx0[n]]
        dx2 = X[5, idx0[n - 1]: idx0[n]]
        dy2 = X[4, idx0[n - 1]: idx0[n]]

        V1 = V[0, idx0[n - 1]: idx0[n]]
        V2 = V[1, idx0[n - 1]: idx0[n]]
        t = T[0, idx0[n - 1]: idx0[n]]

        dist = np.sqrt((dx2 - dx1) ** 2 + ((70 - dy2) - dy1) ** 2)

        pairs = zip(dx1, dy1, dx2, dy2)
        if pointInRect(pairs):
            count += 1
            print(n - 1)
            traj_list.append(n - 1)

    points = np.array([t, dist]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize/Plot Value 1 ColorBar
    lc = LineCollection(segments, cmap='PuOr', norm=norm)  # PuOr, viridis
    if agent1 == True:
        lc.set_array(V1)
    if agent2 == True:
        lc.set_array(V2)
    line = axs.add_collection(lc)

# Configure Plot
axs.set_title(title)

axs.set_xlim(-0.2, 3.2)
axs.set_xlabel("Time")
axs.set_ylim(0, 100)
axs.set_ylabel('Distance between players')
fig.colorbar(line, ax=axs)

for i in range(len(traj_list)):
    dx1 = X0[1, traj_list[i]]
    dy1 = X0[0, traj_list[i]]
    dx2 = X0[5, traj_list[i]]
    dy2 = X0[4, traj_list[i]]
    dist = np.sqrt((dx2 - dx1) ** 2 + ((70 - dy2) - dy1) ** 2)
    axs.plot(dist, marker='o', color='red', markersize=5)

plt.hlines(1.5, 0, 3, color="red", linestyle='--', linewidth=3)

print("Total Collision: %d" %count)
plt.show()


