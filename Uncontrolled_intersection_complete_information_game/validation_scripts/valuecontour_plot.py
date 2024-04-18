import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import math

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}
plt.rc('font', **font)
fig, axs = plt.subplots(1, 1, figsize=(6, 6))

EL = False
HL = True

N = 5
time = np.linspace(0, 3, num=6)
Time = time[N]

if EL == True:
    path = './value/tanh/valuecontour_el_a_a_tanh.mat'
    axs.set_title(r'$\vartheta_{EL}$' + ' contour at t=' + str(round((3 - Time), 2)) + ' s', fontweight='bold')
if HL == True:
    path = './value/tanh/valuecontour_hl_a_a_tanh_old.mat'
    axs.set_title(r'$\vartheta_{HL}$' + ' contour at t=' + str(round((3 - Time), 2)) + ' s', fontweight='bold')

data = scio.loadmat(path)
theta1 = 1
theta2 = 1

d1 = data['d1']
d2 = data['d2']
V1 = data['V1']
V2 = data['V2']

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 3 + theta1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=2, edgecolor='k', facecolor='none')

start1 = patches.Rectangle((15, 15), 5, 5, linewidth=2, edgecolor='k', facecolor='none')
axs.add_patch(train1)
axs.add_patch(start1)
axs.set_xlim(15, 40)
axs.set_xlabel('d1', fontweight='bold')
axs.set_ylim(15, 40)
axs.set_ylabel('d2', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

a = plt.contourf(d1, d2, V1[0, :, :], 3, cmap=plt.cm.Spectral, fontsize=14)
b = plt.contour(d1, d2, V1[0, :, :], 3, colors='black', linewidths=1, linestyles='solid')

file = 'test_data/data_valuecontour_a_a_gt.mat'
data = scio.loadmat(file)

X = data['X']
t = data['t']
V = data['V']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
X0 = X[:, idx0]
V0 = data['V'][0, idx0]
count = 0
index = 2

# collision area
bl = (35 - 1 * 0.75, 35 - 1 * 0.75)
tr = (38.75, 38.75)

V1_collision = []
V2_collision = []
V1_col_index = []
V2_col_index = []

def pointInRect(bl, tr, points):
    for p in points:
        if (p[0] >= bl[0] and p[0] <= tr[0] and p[1] >= bl[1] and p[1] <= tr[1]):
            return True
    return False

for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        x1 = X[0, idx0[n - 1]:]
        x2 = X[index, idx0[n - 1]:]

        V1 = V[0, idx0[n - 1]:]
        V2 = V[1, idx0[n - 1]:]

        pairs = zip(x1, x2)
        if pointInRect(bl, tr, pairs):
            count += 1
            axs.plot(x1[0], x2[0], marker='o', color='orange', markersize=2.5)
            V1_collision.append(V1[0])
            V2_collision.append(V2[0])
            V1_col_index.append(n)
            V2_col_index.append(n)
        else:
            axs.plot(x1[0], x2[0], marker='o', color='black', markersize=2.5)

    else:
        x1 = X[0, idx0[n - 1]: idx0[n]]
        x2 = X[index, idx0[n - 1]: idx0[n]]

        V1 = V[0, idx0[n - 1]: idx0[n]]
        V2 = V[1, idx0[n - 1]: idx0[n]]

        pairs = zip(x1, x2)
        if pointInRect(bl, tr, pairs):
            count += 1
            axs.plot(x1[0], x2[0], marker='o', color='orange', markersize=2.5)
            V1_collision.append(V1[0])
            V2_collision.append(V2[0])
            V1_col_index.append(n)
            V2_col_index.append(n)
        else:
            axs.plot(x1[0], x2[0], marker='o', color='black', markersize=2.5)

V1_collision = np.array(V1_collision).reshape(1, -1)
V2_collision = np.array(V2_collision).reshape(1, -1)
V1_col_index = np.array(V1_col_index).reshape(1, -1)
V2_col_index = np.array(V2_col_index).reshape(1, -1)

plt.show()