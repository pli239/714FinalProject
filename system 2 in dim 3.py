import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ProjectFunction as pf
from mpl_toolkits.mplot3d import Axes3D


# Grid setup
m = 32
mm = m * m * m
h = 1.0 / m
hfac = 1 / (h*h)

num_species = 2
U = [np.zeros((m, m, m)) for _ in range(num_species)]
# for i in range(0, m//2):
#     for j in range(m):
#         U[0][i,j] = 0.5
# for i in range(m//2, m):
#     for j in range(m):
#         U[1][i, j] = 0.5
U[0][0,:, :] = 0.5
U[1][m-1, :, :] = 0.5



a = 1
b = 1
c = 100000
delta_t = h * h * 0.125


# Create the Laplacian matrix
A = np.zeros((mm, mm))
for k in range(m):
    for j in range(m):
        for i in range(m):
            ij = m * m * k + m * j + i
            A[ij, ij] = -4 * hfac
            if i > 0: A[ij, ij - 1] = hfac
            if i < m - 1: A[ij, ij + 1] = hfac
            if j > 0: A[ij, ij - m] = hfac
            if j < m - 1: A[ij, ij + m] = hfac
            if k > 0: A[ij, ij - m * m] = hfac
            if k < m - 1: A[ij, ij + m * m] = hfac


B = np.zeros((mm, mm))
for k in range(m):
    for j in range(m):
        for i in range(m):
            ijk = m * m * k + m * j + i
            if i > 0: B[ijk, ijk - 1] = 1
            if i < m - 1: B[ijk, ijk + 1] = 1
            if j > 0: B[ijk, ijk - m] = 1
            if j < m - 1: B[ijk, ijk + m] = 1
            if k > 0: B[ijk, ijk - m * m] = 1
            if k < m - 1: B[ijk, ijk + m * m] = 1

def Geterror(A, B):
    error = np.sum((A-B)**2) * h**3
    error = error**0.5
    return error

def update_species(U):
    newU = U[:]
    Max = 1
    for i in range(len(U)):
        W = pf.GetW(U, i)
        flag = 0
        while 1:
            laplacian_term = A @ W.flatten()
            F = -laplacian_term - (a * W * (1 - np.abs(W))).flatten()
            JF = A - np.diag((a * (1 - 2 * np.abs(W))).flatten())
            newW = np.linalg.solve(JF, F)
            newW = newW.reshape(m, m, m)
            newW = W - newW
            newW = newW / np.max(np.abs(newW))
            flag += 1
            print("flag=",flag)
            print(Geterror(W, newW))
            print(newW)
            if Geterror(W, newW) < 1e-4:
                break
            W = newW
        newU[i] = newW
        newU[i][newU[i] < 0] = 0
        if Max < np.max(newU[i]):
            Max = np.max(newU[i])
    for i in range(len(U)):
        newU[i] = newU[i] / Max
    return newU

def update_species2(U):
    newU = U[:]
    Max = 1
    for i in range(len(U)):
        W = pf.GetW(U, i)
        WBar = B @ W.flatten()
        WBar = WBar.reshape(m, m, m)
        alpha = a * h * h / 6
        newU[i] = 2 * WBar / (1 - alpha + np.sqrt((1 - alpha)**2 + 4 * alpha * WBar))
        # for j in range(m):
        #     for k in range(m):
        #         if math.isnan(newU[i][j, k]):
        #             newU[i][j, k] = 0
        newU[i][newU[i] < 0] = 0
        if Max < np.max(newU[i]):
            Max = np.max(newU[i])
    for i in range(len(U)):
        newU[i] = newU[i] / Max
    return newU


def Main():
    global U
    mode = 0
    while 1:
        newU = update_species2(U)[:]
        print(mode)
        print(np.sum([Geterror(U[j], newU[j]) for j in range(len(U))], axis=0))
        if np.sum([Geterror(U[j], newU[j]) for j in range(len(U))], axis=0) < 2e-4:
            U = newU[:]
            break
        U = newU[:]
        mode += 1
        if mode - (mode // 100) * 100 == 0:
            # uu = np.zeros((m + 2, m + 2, m + 2))
            # uu[1:m + 1, 1:m + 1, 1:m + 1] = U[0] - U[1]
            # xa = np.linspace(0, 1, m + 2)
            # mgx, mgy, mgz = np.meshgrid(xa, xa, xa)
            # pf.plot_contour(mgx, mgy, uu)
            # pf.plot_3d(mgx, mgy, uu)
            uu = np.ones((m + 2, m + 2))
            K = U[0]
            K[K < 0.1] = 0
            uu[1:m + 1, 1:m + 1] = np.argmin(K, axis=0) - 1
            uu[0,:] = uu[1,:]
            uu[:,0] = uu[:,1]
            uu[m+1,:] = uu[m,:]
            uu[:,m+1] = uu[:,m]
            uu = 1 - uu * h
            xa = np.linspace(0, 1, m + 2)
            mgx, mgy = np.meshgrid(xa, xa)
            pf.plot_3d(mgx, mgy, uu)

            uu = np.zeros((m + 2, m + 2))
            uu[1:m + 1, 1:m + 1] = U[0][:,:,m//2]
            xa = np.linspace(0, 1, m + 2)
            mgx, mgy = np.meshgrid(xa, xa)
            pf.plot_contour(mgx, mgy, uu)


Main()

uu = np.zeros((m + 2, m + 2))
uu[1:m + 1, 1:m + 1] = U[0][:, :, m // 2]
xa = np.linspace(0, 1, m + 2)
mgx, mgy = np.meshgrid(xa, xa)
pf.plot_contour(mgx, mgy, uu)
