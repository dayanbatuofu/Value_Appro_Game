import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

empAAbeliefNANA = True
empNANAbeliefAA = False
nonempAAbeliefNANA = False
nonempNANAbeliefAA = False

agent1 = True
agent2 = False

# collision area
bl = (35 - 1 * 0.75, 35 - 1 * 0.75)
tr = (38.75, 38.75)

def pointInRect(bl, tr, points):
    for p in points:
        if (p[0] >= bl[0] and p[0] <= tr[0] and p[1] >= bl[1] and p[1] <= tr[1]):
            return True
    return False

#________________________________________________________________________________________

if empAAbeliefNANA is True:
    # file = 'inference_model/hybrid/data_E_a_a_belief_na_na.mat'
    file = 'inference_model/supervised/data_E_a_a_belief_na_na.mat'
    data = scipy.io.loadmat(file)
    index = 2
    title = '$(e,e), \Theta^{*}=(a,a), P_0=P_0^{na}$'
    theta1 = 1
    theta2 = 1
if empNANAbeliefAA is True:
    # file = 'inference_model/hybrid/data_E_na_na_belief_a_a.mat'
    file = 'inference_model/supervised/data_E_na_na_belief_a_a.mat'
    data = scipy.io.loadmat(file)
    index = 2
    title = '$(e,e), \Theta^{*}=(na,na), P_0=P_0^{a}$'
    theta1 = 5
    theta2 = 5
if nonempAAbeliefNANA is True:
    # file = 'inference_model/hybrid/data_NE_a_a_belief_na_nar.mat'
    file = 'inference_model/supervised/data_NE_a_a_belief_na_na.mat'
    data = scipy.io.loadmat(file)
    index = 2
    title = '$(ne,ne), \Theta^{*}=(a,a), P_0=P_0^{na}$'
    special = 1
    theta1 = 1
    theta2 = 1
if nonempNANAbeliefAA is True:
    # file = 'inference_model/hybrid/data_NE_na_na_belief_a_a.mat'
    file = 'inference_model/supervised/data_NE_na_na_belief_a_a.mat'
    data = scipy.io.loadmat(file)
    index = 2
    title = '$(ne,ne), \Theta^{*}=(na,na), P_0=P_0^{a}$'
    theta1 = 5
    theta2 = 5

#____________________________________________________________________________________________________

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)

data = scipy.io.loadmat(file)
X = data['X']
P = data['P']
T = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
X0 = X[:, idx0]
count = 0
traj_list = []

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

norm = plt.Normalize(0, 1)

for n in range(1, len(idx0) + 1):
# for n in range(46, 47):
    if n == len(idx0):
        d1 = X[0, idx0[n - 1]:]
        d2 = X[2, idx0[n - 1]:]
        P1 = P[0, idx0[n - 1]:]
        P2 = P[1, idx0[n - 1]:]
        t = T[0, idx0[n - 1]:]

        pairs = zip(d1, d2)
        if pointInRect(bl, tr, pairs):
            count += 1
            print(n - 1)
            traj_list.append(n - 1)
    else:
        d1 = X[0, idx0[n - 1]: idx0[n]]
        d2 = X[2, idx0[n - 1]: idx0[n]]
        P1 = P[0, idx0[n - 1]: idx0[n]]
        P2 = P[1, idx0[n - 1]: idx0[n]]
        t = T[0, idx0[n - 1]: idx0[n]]

        pairs = zip(d1, d2)
        if pointInRect(bl, tr, pairs):
            count += 1
            print(n - 1)
            traj_list.append(n - 1)

    points = np.array([d1, d2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize/Plot Value 1 ColorBar
    lc = LineCollection(segments, cmap='PuOr', norm=norm)  # PuOr, viridis
    if agent1 == True:
        lc.set_array(P1)
    if agent2 == True:
        lc.set_array(P2)
    line = axs.add_collection(lc)

# Configure Plot
axs.set_title(title)

start1 = patches.Rectangle((15, 15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
intersection1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
axs.add_patch(intersection1)
axs.add_patch(start1)

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - 1 * 0.75), 3 + theta1 * 0.75 + 0.75,
                           3 + 1 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
axs.add_patch(train1)
train2 = patches.Rectangle((35 - 1 * 0.75, 35 - theta2 * 0.75), 3 + 1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
axs.add_patch(train2)
axs.set_xlim(15, 40)
axs.set_xlabel('d1')
axs.set_ylim(15, 40)
axs.set_ylabel('d2')
fig.colorbar(line, ax=axs)

for i in range(len(traj_list)):
    axs.plot(X0[0, traj_list[i]], X0[2, traj_list[i]], marker='o', color='red', markersize=5)

print("Total Collision: %d" %count)
plt.show()