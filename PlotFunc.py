import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def format_cont(x, pos):
    return f"{x:.1e}" if x >= 1e4 else f"{x}"


def plot_surface(f, restriction):
    coord_x = np.linspace(restriction[0][0], restriction[1][0], 201)
    coord_y = np.linspace(restriction[0][1], restriction[1][1], 201)
    coord_x_grid, coord_y_grid = np.meshgrid(coord_x, coord_y)
    coord_z_grid = np.zeros(coord_x_grid.shape)
    for i in range(coord_x_grid.shape[0]):
        for j in range(coord_x_grid.shape[1]):
            coord_z_grid[i][j] = f([coord_x_grid[i][j], coord_y_grid[i][j]])

    fig = plt.figure("plot_surface", figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_title('Three-dimensional graph of the function')
    ax.plot_surface(coord_x_grid, coord_y_grid, coord_z_grid, cmap="plasma")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("f ($x_1$,$x_2$)", labelpad=27.5)

    ax.zaxis.set_major_formatter(FuncFormatter(format_cont))

    ax.zaxis.set_tick_params(pad=15)

    ax.set_xlim([restriction[0][0] - 0.5, restriction[1][0] + 0.5])
    ax.set_ylim([restriction[0][1] - 0.5, restriction[1][1] + 0.5])


def contour_plot(f, restriction):
    coord_x = np.linspace(restriction[0][0], restriction[1][0], 201)
    coord_y = np.linspace(restriction[0][1], restriction[1][1], 201)
    coord_x_grid, coord_y_grid = np.meshgrid(coord_x, coord_y)
    coord_z_grid = np.zeros(coord_x_grid.shape)
    for i in range(coord_x_grid.shape[0]):
        for j in range(coord_x_grid.shape[1]):
            coord_z_grid[i][j] = f([coord_x_grid[i][j], coord_y_grid[i][j]])
    fig = plt.figure("contour_plot", figsize=(8, 8))
    cl = plt.contourf(coord_x_grid, coord_y_grid, coord_z_grid, levels=5, cmap="BuPu", extend="min")
    cl2 = plt.contour(coord_x_grid, coord_y_grid, coord_z_grid, colors='k', levels=5)
    plt.clabel(cl2, fmt=FuncFormatter(format_cont), colors="k")
    cbar = fig.colorbar(cl)
    cbar.ax.set_ylabel("Colorbar", rotation=-90, fontsize=12, labelpad=12)
    plt.xlim([restriction[0][0] - 0.1, restriction[1][0] + 0.1])
    plt.ylim([restriction[0][1] - 0.1, restriction[1][1] + 0.1])
    plt.title("Level lines and contour graph")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.tight_layout()
    plt.grid(which='major')