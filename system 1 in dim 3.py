import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ProjectFunction as pf

# Grid setup
m = 32
mm = m * m * m
h = 1.0 / m
hfac = 1 / (h*h)

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


num_species = 2
U = [np.zeros((m, m, m)) for _ in range(num_species)]
U[0][0,:, :] = 0.5
U[1][m-1, :, :] = 0.5
a = 1
b = 1
c = 100000
delta_t = h * h * 0.125


def update_species(U, A, a, b, c, delta_t):
    new_U = [None] * len(U)
    Max = 1
    for i in range(len(U)):
        laplacian_term = A @ U[i].flatten()
        laplacian_term = laplacian_term.reshape(m, m, m)
        interaction_term = np.sum([U[j] for j in range(len(U)) if j != i], axis=0)
        # print(laplacian_term * delta_t)
        new_U[i] = U[i] + delta_t * (laplacian_term + a * U[i] - b * U[i]**2 - c * U[i] * interaction_term)
        new_U[i][new_U[i] < 0] = 0  # Set negative values to zero
        if Max < np.max(new_U[i]):
            Max = np.max(new_U[i])
    for i in range(len(U)):
        new_U[i] = new_U[i] / Max
    return new_U


num_iterations = 100000
for iters in range(num_iterations):
    print(iters)
    U = update_species(U, A, a, b, c, delta_t)

    if iters - (iters // 100) * 100 == 0:
        uu = np.ones((m + 2, m + 2))
        K = U[0]
        K[K < 0.1] = 0
        uu[1:m + 1, 1:m + 1] = np.argmin(K, axis=0) - 1
        uu[0, :] = uu[1, :]
        uu[:, 0] = uu[:, 1]
        uu[m + 1, :] = uu[m, :]
        uu[:, m + 1] = uu[:, m]
        uu = 1 - uu * h
        xa = np.linspace(0, 1, m + 2)
        mgx, mgy = np.meshgrid(xa, xa)
        pf.plot_3d(mgx, mgy, uu)

        uu = np.zeros((m + 2, m + 2))
        uu[1:m + 1, 1:m + 1] = U[0][:, :, m // 2]
        xa = np.linspace(0, 1, m + 2)
        mgx, mgy = np.meshgrid(xa, xa)
        pf.plot_contour(mgx, mgy, uu)

