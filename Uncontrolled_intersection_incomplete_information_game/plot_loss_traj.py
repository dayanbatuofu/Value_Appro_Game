"""
for plotting multiple loss trajectory from a cvs file
"""
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import csv
from matplotlib.collections import LineCollection
import scipy.io

def read_csv_loss():
    # for root,dirs,files in os.walk(directory):
    #     for file in files:
    #         if file.endswith('.csv'):
    #             f = open(file, 'r')
    #             f.close()
    # time = []
    # loss_1 = []
    # loss_2 = []
    # x1 = []
    # x2 = []
    t_OUT = np.empty((1, 0))
    X_OUT = np.empty((4, 0))
    U_OUT = np.empty((2, 0))
    P_OUT = np.empty((2, 0))

    "choose path here"
    # path = "experiment/bvp_empathetic A_A belief A_A/*.csv"
    path = "experiment/bvp_non_empathetic A_A belief A_A/*.csv"
    # path = "experiment/bvp_empathetic A_A belief NA_NA/*.csv"
    # path = "experiment/bvp_non_empathetic A_A belief NA_NA/*.csv"
    # path = "experiment/bvp_empathetic NA_NA belief NA_NA/*.csv"
    # path = "experiment/bvp_non_empathetic NA_NA belief NA_NA/*.csv"
    # path = "experiment/bvp_empathetic NA_NA belief A_A/*.csv"
    # path = "experiment/bvp_non_empathetic NA_NA belief A_A/*.csv"

    for filename in glob.glob(path):
        print(filename)
        with open(filename, 'r') as csv_file:
            # creating a csv reader object
            csv_reader = csv.reader(csv_file)

            # extracting each data row one by one
            rows = []
            for row in csv_reader:
                rows.append(row)
            result = np.array(rows[0])
            t_OUT = np.hstack((t_OUT, np.array(rows[0], dtype=np.float32).reshape(1, -1)))
            X_OUT = np.hstack((X_OUT, np.vstack((np.array(rows[3], dtype=np.float32),
                                                 np.array(rows[4], dtype=np.float32),
                                                 np.array(rows[5], dtype=np.float32),
                                                 np.array(rows[6], dtype=np.float32)))))
            U_OUT = np.hstack((U_OUT, np.vstack((np.array(rows[1][1:], dtype=np.float32),
                                                 np.array(rows[2][1:], dtype=np.float32)))))
            P_OUT = np.hstack((P_OUT, np.vstack((np.array(rows[7], dtype=np.float32),
                                                 np.array(rows[8], dtype=np.float32)))))
    dataset = dict()
    dataset.update({'t': t_OUT,
                    'X': X_OUT,
                    'U': U_OUT,
                    'P': P_OUT})

    # save_path = "experiment/bvp_empathetic A_A belief A_A/data_E_a_a_belief_a_a.mat"
    save_path = "experiment/bvp_non_empathetic A_A belief A_A/data_NE_a_a_belief_a_a.mat"
    # save_path = "experiment/bvp_empathetic A_A belief NA_NA/data_E_a_a_belief_na_na.mat"
    # save_path = "experiment/bvp_non_empathetic A_A belief NA_NA/data_NE_a_a_belief_na_na.mat"
    # save_path = "experiment/bvp_empathetic NA_NA belief NA_NA/data_E_na_na_belief_na_na.mat"
    # save_path = "experiment/bvp_non_empathetic NA_NA belief NA_NA/data_NE_na_na_belief_na_na.mat"
    # save_path = "experiment/bvp_empathetic NA_NA belief A_A/data_E_na_na_belief_a_a.mat"
    # save_path = "experiment/bvp_non_empathetic NA_NA belief A_A/data_NE_na_na_belief_a_a.mat"

    scipy.io.savemat(save_path, dataset)

    # return loss_1, loss_2, x1, x2


def plot_loss():
    loss_s, x1_s, x2_s = read_csv_loss()
    loss = []
    x1 = []
    x2 = []
    for i in range(len(loss_s)):
        loss.append([float(j) for j in loss_s[i]])
        x1.append([float(j) for j in x1_s[i]])
        x2.append([float(j) for j in x2_s[i]])
    # print(loss)
    # plt.show()
    n = len(loss)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        print(len(x1[i]), len(x2[i]), len(loss[i]))
        ax.scatter(x1[i], x2[i], loss[i])
    # ax = fig.add_subplot(111, projection='3d')
    # print(len(x1[0]), len(x2[0]), len(loss[0]))
    # print(x1[0][0])
    # ax.scatter(x1[0], x2[0], loss[0])
    ax.invert_xaxis()
    ax.set_xlabel('P1 location')
    ax.set_ylabel('P2 location')
    ax.set_zlabel('Loss')
    ax.set_xticks([15, 20, 25, 30, 35, 40, 45])
    ax.set_yticks([15, 20, 25, 30, 35, 40, 45])
    # ax.xlim([15, 40])
    # ax.ylim([15, 40])
    # ax.axis('equal')
    plt.show()


def plot_loss_color():
    loss_s, x1_s, x2_s = read_csv_loss()
    loss = []
    x1 = []
    x2 = []
    max_loss = 0
    min_loss = 0
    for i in range(len(loss_s)):
        loss.append([float(j) for j in loss_s[i]])
        x1.append([float(j) for j in x1_s[i]])
        x2.append([float(j) for j in x2_s[i]])
        if max(loss[i]) > max_loss:
            max_loss = max(loss[i])
        if min(loss[i])<min_loss:
            min_loss = min(loss[i])
    # print(loss)
    # plt.show()
    n = len(loss)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(min_loss, max_loss)
    for i in range(n):
        # print(len(x1[i]), len(x2[i]), len(loss[i]))
        # x = np.zeros((len(x1[i]), 2))
        # for j in range(len(x1[i])):
        #     x[j][0] = x1[i][j]
        #     x[j][1] = x2[i][j]
        x = np.column_stack((x1[i], x2[i]))
        print(x)
        lc = LineCollection(x, cmap='viridis', norm=norm)
        lc.set_array(loss[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        # ax.scatter(x1[i], x2[i], loss[i])
    # ax = fig.add_subplot(111, projection='3d')
    # print(len(x1[0]), len(x2[0]), len(loss[0]))
    # print(x1[0][0])
    # ax.scatter(x1[0], x2[0], loss[0])
    ax.invert_xaxis()
    ax.set_xlabel('P1 location')
    ax.set_ylabel('P2 location')
    # ax.set_zlabel('Loss')
    fig.colorbar(line, ax=ax[0])
    ax.set_xticks([15, 20, 25, 30, 35, 40, 45])
    ax.set_yticks([15, 20, 25, 30, 35, 40, 45])
    # ax.xlim([15, 40])
    # ax.ylim([15, 40])
    # ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    # plot_loss()
    read_csv_loss()
