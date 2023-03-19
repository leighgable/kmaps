import matplotlib.pyplot as plt
from matplotlib import ticker

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d

def plot_3d(points, points_color, title, res_file_name="test_3d.png"):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.savefig(res_file_name)
    
def plot_2d(points, points_color, title, res_file_name="test_2d.png"):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.savefig(res_file_name)
    
def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    
# Example of using add_2d_scatter to subplot
# fig, axs = plt.subplots(
#     nrows=2, ncols=2, figsize=(7, 7), facecolor="white", constrained_layout=True
# )
# fig.suptitle("Locally Linear Embeddings", size=16)

# lle_methods = [
#     ("Standard locally linear embedding", S_standard),
#     ("Local tangent space alignment", S_ltsa),
#     ("Hessian eigenmap", S_hessian),
#     ("Modified locally linear embedding", S_mod),
# ]
# for ax, method in zip(axs.flat, lle_methods):
#     name, points = method
#     add_2d_scatter(ax, points, S_color, name)
