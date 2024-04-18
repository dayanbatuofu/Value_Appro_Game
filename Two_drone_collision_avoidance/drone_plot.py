import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io
from math import cos, sin

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 24}

plt.rc('font', **font)

load_path = 'examples/vehicle/data_test_drone_GT_5.mat'
# load_path = 'examples/vehicle/data_test_drone_HL_5.mat'
# load_path = 'examples/vehicle/data_test_drone_SL_5.mat'
# load_path = 'examples/vehicle/data_test_drone_SSL_5.mat'
data = scipy.io.loadmat(load_path)
data.update({'t0': data['t']})
X = data['X']
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]
n = 0

s1x = data['X'][0, idx0[n]:]
s1y = data['X'][1, idx0[n]:]
s1z = data['X'][2, idx0[n]:]

s2x = data['X'][6, idx0[n]:]
s2y = data['X'][7, idx0[n]:]
s2z = data['X'][8, idx0[n]:]

roll1 = data['Theta'][0, idx0[n]:]
roll2 = data['Theta'][1, idx0[n]:]
pitch1 = data['Phi'][0, idx0[n]:]
pitch2 = data['Phi'][1, idx0[n]:]
yaw1 = 0
yaw2 = 0

dist = np.sqrt(((5 - s2x) - s1x) ** 2 + ((5 - s2y) - s1y) ** 2 + (s2z - s1z) ** 2)

R1 = np.array(5)
R2 = np.array(5)

p1 = np.array([0.45, 0, 0, 1]).T
p2 = np.array([-0.45, 0, 0, 1]).T
p3 = np.array([0, 0.45, 0, 1]).T
p4 = np.array([0, -0.45, 0, 1]).T

def transform_maxtrix(yaw, pitch, roll, x, y, z):
    T_matrix = np.array([[cos(yaw) * cos(pitch), -sin(yaw1) * cos(roll) + cos(yaw1) * sin(pitch) * sin(roll),
                          sin(yaw) * sin(roll) + cos(yaw1) * sin(pitch) * cos(roll), x],
                         [sin(yaw) * cos(pitch), cos(yaw1) * cos(roll) + sin(yaw1) * sin(pitch) * sin(roll),
                          -cos(yaw) * sin(roll) + sin(yaw1) * sin(pitch) * cos(roll), y],
                         [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw1), z]])
    return T_matrix

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.plot3D(s1x, s1y, s1z, linewidth=2.5, linestyle='-.', label='Player 1 trajectory', c='steelblue')
ax.plot3D(R1-s2x, R2-s2y, s2z, linewidth=2.5, linestyle='-.', label='Player 2 trajectory', c='limegreen')

ax.scatter3D(s1x[0], s1y[0], s1z[0], s=300, marker='*', label='start point of Player 1', c='steelblue')
ax.scatter3D(R1-s2x[0], R2-s2y[0], s2z[0], s=300, marker='*', label='start point of Player 2', c='limegreen')

# create drone for initial state
T_matrix = transform_maxtrix(yaw1, pitch1[0], roll1[0], s1x[0], s1y[0], s1z[0])

p0_11 = np.matmul(T_matrix, p1)
p0_12 = np.matmul(T_matrix, p2)
p0_13 = np.matmul(T_matrix, p3)
p0_14 = np.matmul(T_matrix, p4)

ax.plot([p0_11[0], p0_12[0], p0_13[0], p0_14[0]],
        [p0_11[1], p0_12[1], p0_13[1], p0_14[1]],
        [p0_11[2], p0_12[2], p0_13[2], p0_14[2]], 'k.', markersize=10)
ax.plot([p0_11[0], p0_12[0]], [p0_11[1], p0_12[1]],
        [p0_11[2], p0_12[2]], 'r-', linewidth=3)
ax.plot([p0_13[0], p0_14[0]], [p0_13[1], p0_14[1]],
        [p0_13[2], p0_14[2]], 'r-', linewidth=3)

T_matrix = transform_maxtrix(yaw2, pitch2[0], roll2[0], R1-s2x[0], R2-s2y[0], s2z[0])

p0_21 = np.matmul(T_matrix, p1)
p0_22 = np.matmul(T_matrix, p2)
p0_23 = np.matmul(T_matrix, p3)
p0_24 = np.matmul(T_matrix, p4)

ax.plot([p0_21[0], p0_22[0], p0_23[0], p0_24[0]],
        [p0_21[1], p0_22[1], p0_23[1], p0_24[1]],
        [p0_21[2], p0_22[2], p0_23[2], p0_24[2]], 'k.', markersize=10)
ax.plot([p0_21[0], p0_22[0]], [p0_21[1], p0_22[1]],
        [p0_21[2], p0_22[2]], 'r-', linewidth=3)
ax.plot([p0_23[0], p0_24[0]], [p0_23[1], p0_24[1]],
        [p0_23[2], p0_24[2]], 'r-', linewidth=3)

# create drone for final state
T_matrix = transform_maxtrix(yaw1, pitch1[-1], roll1[-1], s1x[-1], s1y[-1], s1z[-1])

pT_11 = np.matmul(T_matrix, p1)
pT_12 = np.matmul(T_matrix, p2)
pT_13 = np.matmul(T_matrix, p3)
pT_14 = np.matmul(T_matrix, p4)

ax.plot([pT_11[0], pT_12[0], pT_13[0], pT_14[0]],
        [pT_11[1], pT_12[1], pT_13[1], pT_14[1]],
        [pT_11[2], pT_12[2], pT_13[2], pT_14[2]], 'k.', markersize=10)
ax.plot([pT_11[0], pT_12[0]], [pT_11[1], pT_12[1]],
        [pT_11[2], pT_12[2]], 'r-', linewidth=3)
ax.plot([pT_13[0], pT_14[0]], [pT_13[1], pT_14[1]],
        [pT_13[2], pT_14[2]], 'r-', linewidth=3)

T_matrix = transform_maxtrix(yaw2, pitch2[-1], roll2[-1], R1-s2x[-1], R2-s2y[-1], s2z[-1])

pT_21 = np.matmul(T_matrix, p1)
pT_22 = np.matmul(T_matrix, p2)
pT_23 = np.matmul(T_matrix, p3)
pT_24 = np.matmul(T_matrix, p4)

ax.plot([pT_21[0], pT_22[0], pT_23[0], pT_24[0]],
        [pT_21[1], pT_22[1], pT_23[1], pT_24[1]],
        [pT_21[2], pT_22[2], pT_23[2], pT_24[2]], 'k.', markersize=10)
ax.plot([pT_21[0], pT_22[0]], [pT_21[1], pT_22[1]],
        [pT_21[2], pT_22[2]], 'r-', linewidth=3)
ax.plot([pT_23[0], pT_24[0]], [pT_23[1], pT_24[1]],
        [pT_23[2], pT_24[2]], 'r-', linewidth=3)

# title = 'Two-drone Collision Avoidance Games'
# ax.set_title(title, fontweight='bold')

# ax.set_xlabel('X(m)', labelpad=10)
# ax.set_ylabel('Y(m)', labelpad=20)
# ax.set_zlabel('Z(m)', labelpad=15)
# plt.legend(bbox_to_anchor=(0.3, 0.08), loc='upper right', borderaxespad=0)

ax.set_xlabel('X(m)', labelpad=14, fontweight='bold')
ax.set_ylabel('Y(m)', labelpad=18, fontweight='bold')
ax.set_zlabel('Z(m)', labelpad=20, fontweight='bold')
plt.legend(bbox_to_anchor=(1.12, 1.15), borderaxespad=0)

# only for SSL
# ax.set_zlim(-1, 1)

# paper visualization plot
# ax.view_init(20, 140)
# filename = "./" + str(120) + ".png"
# plt.savefig(filename)
# print("Save " + filename + " finish")

# animation plot
for angle in range(90, 183, 1):
    ax.view_init(30, angle)
    filename = 'drone_case/drone_' + str(angle) + '.png'
    plt.savefig(filename)
    print("Save " + filename + " finish")

# ax.set_xlabel('X(m)', fontweight='bold')
# ax.set_ylabel('Y(m)', fontweight='bold')
# ax.set_zlabel('Z(m)', fontweight='bold')
# plt.xticks(fontweight='bold')
# plt.yticks(fontweight='bold')
# plt.legend(prop={'weight': 'bold'}, loc='best')

plt.show()

