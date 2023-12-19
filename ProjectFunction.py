import numpy as np
from matplotlib import pyplot as plt, cm


def From2dTo1d(i, j, m):
    return m * i + j

def From1dTo2d(ij, m):
    j = ij//m
    i = ij - m * j
    return i, j


def GetW(U, i):
    return U[i] - np.sum([U[j] for j in range(len(U)) if j != i], axis=0)


def plot_contour(X, Y, U):
    plt.contour(X, Y, U)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot')
    plt.colorbar(label='U values')
    plt.show()

def plot_3d(X, Y, U):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, U, cmap=cm.plasma, rstride=1, cstride=1, linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('U')
    plt.show()