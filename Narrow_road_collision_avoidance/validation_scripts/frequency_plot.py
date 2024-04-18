import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection

intersection = True
avoidance = False
lane = False
HL = True
VH = False
N_index = 3
N_input = 0

file_hl = 'value/tanh/variable=' + str(N_input) + '/fft_value_hybrid_HD_tanh.mat'
file_vh = 'value/tanh/variable=' + str(N_input) + '/rev0/fft_value_valuehardening_HD_tanh.mat'
t_final = 3
if HL == True:
    title = 'Case 2 $F[V]_{HL}$ at t=' + str(t_final - N_index) + 's'
if VH == True:
    title = 'Case 2 $F[V]_{VH}$ at t=' + str(t_final - N_index) + 's'

data_hl = scipy.io.loadmat(file_hl)
fft_V_hl = data_hl['fft_V1'][N_index, :, :]

data_vh = scipy.io.loadmat(file_vh)
fft_V_vh = data_vh['fft_V1'][N_index, :, :]

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}
plt.rc('font', **font)
fig, axs = plt.subplots(1, 1, figsize=(8, 6))

if HL == True:
    plt.imshow(fft_V_hl)
if VH == True:
    plt.imshow(fft_V_vh)

axs.set_title(title, fontweight='bold')
axs.set_xlabel('Frequency index in p1x', fontweight='bold')
axs.set_ylabel('Frequency index in p2x', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
cbar = plt.colorbar()
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight('bold')

plt.show()
