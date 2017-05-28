from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import mpl_toolkits.mplot3d
import sklearn
import sklearn.decomposition

rnd.seed(42)
PROJECT_ROOT_DIR = os.getcwd() + "/Dementionality_Reduction"

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def save_fig(fig_id, tight_layout=True):
    """
    Save the previous plotted image as .png file.
    :param fig_id: name of the image
    :param tight_layout: Automatically adjust subplot parameters to give specified padding.
    :return: 
    """
    assert isinstance(fig_id, str), "Must be a string."
    if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, "images")):
        os.makedirs(os.path.join(PROJECT_ROOT_DIR, "images"))
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure: ", "<", fig_id, ">")
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format="png", dpi=300)


#
# # Projection methods
#
# Build 3D dataset:

rnd.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5  # 60个-0.5到3Pi/2的随机弧度
X = np.empty((m, 3))  # 生成60行3列的空矩阵
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * rnd.randn(m) / 2  # 第一列
X[:, 1] = np.sin(angles) + 0.7 + noise * rnd.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * rnd.randn(m)

# Mean normalize the data_
X = X - X.mean(axis=0)  # 每列中的各元素减去所在列的平均值

# Apply PCA to reduce to 2D.
pca = sklearn.decomposition.PCA(n_components=2)
X2D = pca.fit_transform(X)

# Recover the 3D points projected on the plane (PCA 2D subspace).
X2D_inv = pca.inverse_transform(X2D)


# Utility class to draw 3D arrows
class Arrow3D(matplotlib.patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        matplotlib.patches.FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        matplotlib.patches.FancyArrowPatch.draw(self, renderer)


# Express the plane as a function of x and y.
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[2], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

# Plot the 3D dataset, the plane and the projections on that plane.
fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection="3d")

X3D_above = X[X[:, 2] > X2D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X2D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")

np.linalg.norm(C, axis=0)
ax.add_artist(Arrow3D(
    [0, C[0, 0]],
    [0, C[0, 1]],
    [0, C[0, 2]],
    mutation_scale=15,
    linewidth=1,
    arrowstyle="-|>",
    color="k"
))
ax.add_artist(Arrow3D(
    [0, C[1, 0]],
    [0, C[1, 1]],
    [0, C[1, 2]],
    mutation_scale=15,
    linewidth=1,
    arrowstyle="-|>",
    color="k"
))
ax.plot([0], [0], [0], "k")

for i in range(m):
    if X[i, 2] > X2D_inv[i, 2]:
        ax.plot(
            [X[i][0], X2D_inv[i][0]],
            [X[i][1], X2D_inv[i][1]],
            [X[i][2], X2D_inv[i][2]],
            "k-"
        )
    else:
        ax.plot(
            [X[i][0], X2D_inv[i][0]],
            [X[i][1], X2D_inv[i][1]],
            [X[i][2], X2D_inv[i][2]],
            "k-",
            color="#505050"
        )

ax.plot(X2D_inv[:, 0], X2D_inv[:, 1], X2D_inv[:, 2], "k+")
ax.plot(X2D_inv[:, 0], X2D_inv[:, 1], X2D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("dataset_3d_plot")
