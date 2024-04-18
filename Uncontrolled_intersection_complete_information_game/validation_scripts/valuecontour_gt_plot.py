import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# from examples.choose_problem import system

# ____________________________________________________________________________________________________
bvpAA = True
bvpANA = False
bvpNAA = False
bvpNANA = False
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

# ________________________________________________________________________________________
file = 'test_data/data_valuecontour_a_a_gt.mat'
title = 'GT $\u03B8=(a,a)$'
index = 2
theta1 = 1
theta2 = 1
# ____________________________________________________________________________________________________

data = scipy.io.loadmat(file)

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}
plt.rc('font', **font)

X = data['X']
t = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
X0 = X[:, idx0]
count = 0

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        x1 = X[0, idx0[n - 1]:]
        x2 = X[index, idx0[n - 1]:]

        pairs = zip(x1, x2)
        if pointInRect(bl, tr, pairs):
            count += 1
            axs.plot(x1[0], x2[0], marker='o', color='orange', markersize=2.5)
        else:
            axs.plot(x1[0], x2[0], marker='o', color='black', markersize=2.5)

    else:
        x1 = X[0, idx0[n - 1]: idx0[n]]
        x2 = X[index, idx0[n - 1]: idx0[n]]

        pairs = zip(x1, x2)
        if pointInRect(bl, tr, pairs):
            count += 1
            axs.plot(x1[0], x2[0], marker='o', color='orange', markersize=2.5)
        else:
            axs.plot(x1[0], x2[0], marker='o', color='black', markersize=2.5)

# Configure Plot
axs.set_title(title, fontweight='bold')

start1 = patches.Rectangle((15, 15), 5, 5, linewidth=2, edgecolor='k', facecolor='none')
intersection1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
axs.add_patch(intersection1)
axs.add_patch(start1)

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - 1 * 0.75), 3 + theta1 * 0.75 + 0.75,
                           3 + 1 * 0.75 + 0.75, linewidth=2, edgecolor='k', facecolor='none')
axs.add_patch(train1)
train2 = patches.Rectangle((35 - 1 * 0.75, 35 - theta2 * 0.75), 3 + 1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=2, edgecolor='k', facecolor='none')
axs.add_patch(train2)
axs.set_xlim(15, 40)
axs.set_xlabel('d1', fontweight='bold')
axs.set_ylim(15, 40)
axs.set_ylabel('d2', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

print("Total Collision: %d" %count)

plt.show()

