import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# from examples.choose_problem import system

# ____________________________________________________________________________________________________
bvpAA = False
bvpNANA = False
empAAbeliefAA = True
empNANAbeliefNANA = False
nonempAAbeliefAA = False
nonempNANAbeliefNANA = False

# collision area
bl = (35 - 1 * 0.75, 35 - 1 * 0.75)
tr = (38.75, 38.75)

def pointInRect(bl, tr, points):
    for p in points:
        if (p[0] >= bl[0] and p[0] <= tr[0] and p[1] >= bl[1] and p[1] <= tr[1]):
            return True
    return False

# ________________________________________________________________________________________
if bvpAA is True:
	file = 'data_a_a_infer_150.mat'
	index = 2
	title = 'GT $\Theta^{*}=(a,a)$'
	special = 0
	theta1 = 1
	theta2 = 1

if bvpNANA is True:
	file = 'data_na_na_infer_150_18.mat'
	index = 2
	title = 'GT $\Theta^{*}=(na,na)$'
	special = 0
	theta1 = 5
	theta2 = 5

if empAAbeliefAA is True:
	# file = 'inference_model/hybrid/data_E_a_a_belief_a_a.mat'
	file = 'inference_model/supervised/data_E_a_a_belief_a_a.mat'
	index = 2
	title = '$(e,e), \Theta^{*}=(a,a), P_0=P_0^{a}$'
	special = 0
	theta1 = 1
	theta2 = 1

if empNANAbeliefNANA is True:
	# file = 'inference_model/hybrid/data_E_na_na_belief_na_na.mat'
	file = 'inference_model/supervised/data_E_na_na_belief_na_na.mat'
	index = 2
	title = '$(e,e), \Theta^{*}=(na,na), P_0=P_0^{na}$'
	special = 0
	theta1 = 5
	theta2 = 5

if nonempAAbeliefAA is True:
	# file = 'inference_model/hybrid/data_NE_a_a_belief_a_a.mat'
	file = 'inference_model/supervised/data_NE_a_a_belief_a_a.mat'
	index = 2
	title = '$(ne,ne), \Theta^{*}=(a,a), P_0=P_0^{a}$'
	special = 0
	theta1 = 1
	theta2 = 1

if nonempNANAbeliefNANA is True:
	# file = 'inference_model/hybrid/data_NE_na_na_belief_na_na.mat'
	file = 'inference_model/supervised/data_NE_na_na_belief_na_na.mat'
	index = 2
	title = '$(ne,ne), \Theta^{*}=(na,na), P_0=P_0^{na}$'
	special = 0
	theta1 = 5
	theta2 = 5
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
traj_list = []

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

for n in range(1, len(idx0) + 1):
	if n == len(idx0):
		x1 = X[0, idx0[n - 1]:]
		x2 = X[index, idx0[n - 1]:]
		axs.plot(x1, x2, c='black')  # 0.8

		pairs = zip(x1, x2)
		if pointInRect(bl, tr, pairs):
			count += 1
			print(n - 1)
			traj_list.append(n - 1)

	else:
		x1 = X[0, idx0[n - 1]: idx0[n]]
		x2 = X[index, idx0[n - 1]: idx0[n]]
		axs.plot(x1, x2, c='black')  # 0.8

		pairs = zip(x1, x2)
		if pointInRect(bl, tr, pairs):
			count += 1
			print(n - 1)
			traj_list.append(n - 1)

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

for i in range(len(traj_list)):
    axs.plot(X0[0, traj_list[i]], X0[2, traj_list[i]], marker='o', color='red', markersize=5)

print("Total Collision: %d" %count)

plt.show()
