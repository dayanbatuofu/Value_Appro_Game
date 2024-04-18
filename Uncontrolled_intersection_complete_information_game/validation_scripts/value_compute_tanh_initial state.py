'''
Run this script to check the difference between real Value and Value from costate/value network
'''

import numpy as np
import scipy.io

bvpAA = True
bvpNANA = False
bvpANA = False
bvpNAA = False

if bvpAA is True:
    path1 = 'test_data/data_test_a_a_600_18.mat'
    path2 = 'value/tanh/value_generation_hybrid_initial_a_a_tanh.mat'
    path3 = 'value/tanh/value_generation_supervised_initial_a_a_tanh.mat'
    path4 = 'value/tanh/value_generation_selfsupervised_initial_a_a_tanh.mat'
    path5 = 'value/tanh/value_generation_valuehardening_initial_a_a_tanh.mat'
if bvpNANA is True:
    path1 = 'test_data/data_test_na_na_600_18.mat'
    path2 = 'value/tanh/value_generation_hybrid_initial_na_na_tanh.mat'
    path3 = 'value/tanh/value_generation_supervised_initial_na_na_tanh.mat'
    path4 = 'value/tanh/value_generation_selfsupervised_initial_na_na_tanh.mat'
    path5 = 'value/tanh/value_generation_valuehardening_initial_na_na_tanh.mat'
if bvpANA is True:
    path1 = 'test_data/data_test_a_na_600_18.mat'
    path2 = 'value/tanh/value_generation_hybrid_initial_a_na_tanh.mat'
    path3 = 'value/tanh/value_generation_supervised_initial_a_na_tanh.mat'
    path4 = 'value/tanh/value_generation_selfsupervised_initial_a_na_tanh.mat'
    path5 = 'value/tanh/value_generation_valuehardening_initial_a_na_tanh.mat'
if bvpNAA is True:
    path1 = 'test_data/data_test_na_a_600_18.mat'
    path2 = 'value/tanh/value_generation_hybrid_initial_na_a_tanh.mat'
    path3 = 'value/tanh/value_generation_supervised_initial_na_a_tanh.mat'
    path4 = 'value/tanh/value_generation_selfsupervised_initial_na_a_tanh.mat'
    path5 = 'value/tanh/value_generation_valuehardening_initial_na_a_tanh.mat'

data1 = scipy.io.loadmat(path1)
data2 = scipy.io.loadmat(path2)
data3 = scipy.io.loadmat(path3)
data4 = scipy.io.loadmat(path4)
data5 = scipy.io.loadmat(path5)

V_real = data1['V']
V_pred_hybrid = data2['V']
V_pred_supervised = data3['V']
V_pred_selfsupervised = data4['V']
V_pred_valuehardening = data5['V']

data1.update({'t0': data1['t']})
idx0 = np.nonzero(np.equal(data1.pop('t0'), 0.))[1]

N = 151

V1_error_hybrid = np.zeros((len(idx0), N))
V2_error_hybrid = np.zeros((len(idx0), N))
V1_error_supervised = np.zeros((len(idx0), N))
V2_error_supervised = np.zeros((len(idx0), N))
V1_error_selfsupervised = np.zeros((len(idx0), N))
V2_error_selfsupervised = np.zeros((len(idx0), N))
V1_error_valuehardening = np.zeros((len(idx0), N))
V2_error_valuehardening = np.zeros((len(idx0), N))

for i in range(1, len(idx0) + 1):
    if i == len(idx0):
        for j in range(N):
            V1_error_hybrid[i - 1][j] = np.abs(V_real[0, idx0[i - 1]:][j] - V_pred_hybrid[0, idx0[i - 1]:][j])
            V2_error_hybrid[i - 1][j] = np.abs(V_real[1, idx0[i - 1]:][j] - V_pred_hybrid[1, idx0[i - 1]:][j])
            V1_error_supervised[i - 1][j] = np.abs(V_real[0, idx0[i - 1]:][j] - V_pred_supervised[0, idx0[i - 1]:][j])
            V2_error_supervised[i - 1][j] = np.abs(V_real[1, idx0[i - 1]:][j] - V_pred_supervised[1, idx0[i - 1]:][j])
            V1_error_selfsupervised[i - 1][j] = np.abs(V_real[0, idx0[i - 1]:][j] - V_pred_selfsupervised[0, idx0[i - 1]:][j])
            V2_error_selfsupervised[i - 1][j] = np.abs(V_real[1, idx0[i - 1]:][j] - V_pred_selfsupervised[1, idx0[i - 1]:][j])
            V1_error_valuehardening[i - 1][j] = np.abs(V_real[0, idx0[i - 1]:][j] - V_pred_valuehardening[0, idx0[i - 1]:][j])
            V2_error_valuehardening[i - 1][j] = np.abs(V_real[1, idx0[i - 1]:][j] - V_pred_valuehardening[1, idx0[i - 1]:][j])
    else:
        for j in range(N):
            V1_error_hybrid[i - 1][j] = np.abs(V_real[0, idx0[i - 1]: idx0[i]][j] - V_pred_hybrid[0, idx0[i - 1]: idx0[i]][j])
            V2_error_hybrid[i - 1][j] = np.abs(V_real[1, idx0[i - 1]: idx0[i]][j] - V_pred_hybrid[1, idx0[i - 1]: idx0[i]][j])
            V1_error_supervised[i - 1][j] = np.abs(V_real[0, idx0[i - 1]: idx0[i]][j] - V_pred_supervised[0, idx0[i - 1]: idx0[i]][j])
            V2_error_supervised[i - 1][j] = np.abs(V_real[1, idx0[i - 1]: idx0[i]][j] - V_pred_supervised[1, idx0[i - 1]: idx0[i]][j])
            V1_error_selfsupervised[i - 1][j] = np.abs(V_real[0, idx0[i - 1]: idx0[i]][j] - V_pred_selfsupervised[0, idx0[i - 1]: idx0[i]][j])
            V2_error_selfsupervised[i - 1][j] = np.abs(V_real[1, idx0[i - 1]: idx0[i]][j] - V_pred_selfsupervised[1, idx0[i - 1]: idx0[i]][j])
            V1_error_valuehardening[i - 1][j] = np.abs(V_real[0, idx0[i - 1]: idx0[i]][j] - V_pred_valuehardening[0, idx0[i - 1]: idx0[i]][j])
            V2_error_valuehardening[i - 1][j] = np.abs(V_real[1, idx0[i - 1]: idx0[i]][j] - V_pred_valuehardening[1, idx0[i - 1]: idx0[i]][j])

V_mae_hybrid = np.mean((np.hstack((V1_error_hybrid, V2_error_hybrid))))
V_var_hybrid = np.var((np.hstack((V1_error_hybrid, V2_error_hybrid))))

V_mae_supervised = np.mean((np.hstack((V1_error_supervised, V2_error_supervised))))
V_var_supervised = np.var((np.hstack((V1_error_supervised, V2_error_supervised))))

V_mae_selfsupervised = np.mean((np.hstack((V1_error_selfsupervised, V2_error_selfsupervised))))
V_var_selfsupervised = np.var((np.hstack((V1_error_selfsupervised, V2_error_selfsupervised))))

V_mae_valuehardening = np.mean((np.hstack((V1_error_valuehardening, V2_error_valuehardening))))
V_var_valuehardening = np.var((np.hstack((V1_error_valuehardening, V2_error_valuehardening))))

print("MAE of hybrid: %f" % V_mae_hybrid)
print("Var of hybrid: %f" % V_var_hybrid)
print("MAE of supervised: %f" % V_mae_supervised)
print("Var of supervised: %f" % V_var_supervised)
print("MAE of self-supervised: %f" % V_mae_selfsupervised)
print("Var of self-supervised: %f" % V_var_selfsupervised)
print("MAE of value hardening: %f" % V_mae_valuehardening)
print("Var of value hardening: %f" % V_var_valuehardening)