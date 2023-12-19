import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ProjectFunction as pf

# Grid setup
m = 128
mm = m * m
h = 1.0 / m
hfac = 1 / (h*h)

# Create the Laplacian matrix
A = np.zeros((mm, mm))
for i in range(m):
    for j in range(m):
        ij = m * j + i
        A[ij, ij] = -4 * hfac
        if i > 0: A[ij, ij - 1] = hfac
        if i < m - 1: A[ij, ij + 1] = hfac
        if j > 0: A[ij, ij - m] = hfac
        if j < m - 1: A[ij, ij + m] = hfac




num_species = 4
U = [np.zeros((m, m)) for _ in range(num_species)]
U[0][0,:] = 0.5
U[1][m-1, :] = 0.5
U[2][1:m-2,0] = 0.5
U[3][1:m-2,m-1] = 0.5
a = 1
b = 1
c = 100000
delta_t = h * h * 0.125


def update_species(U, A, a, b, c, delta_t):
    new_U = [None] * len(U)
    Max = 0
    for i in range(len(U)):
        laplacian_term = A @ U[i].flatten()
        laplacian_term = laplacian_term.reshape(m, m)
        interaction_term = np.sum([U[j] for j in range(len(U)) if j != i], axis=0)
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
        print(U[0] - U[1])
        uu = np.zeros((m + 2, m + 2))
        uu[1:m + 1, 1:m + 1] = U[0] - U[1]

        xa = np.linspace(0, 1, m + 2)
        mgx, mgy = np.meshgrid(xa, xa)
        pf.plot_3d(mgx,mgy,uu)
        pf.plot_contour(mgx,mgy,uu)

        uu = np.zeros((m + 2, m + 2))
        uu[1:m + 1, 1:m + 1] = U[2] - U[3]

        xa = np.linspace(0, 1, m + 2)
        mgx, mgy = np.meshgrid(xa, xa)
        pf.plot_3d(mgx, mgy, uu)
        pf.plot_contour(mgx, mgy, uu)





