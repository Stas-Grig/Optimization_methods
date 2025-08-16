import numpy as np
from LinearMinimaze import binary_search, golden_search, goldstein_armijo
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def gradient_methods(func, x_0, limit, name_method="CG_PR2", name_linear_method="GS", eps=1e-3):
    def format_cont(x, pos):
        return f"{x:.1e}" if x >= 1e4 else f"{x}"

    def contour_plot(f, restriction, x, k):
        coord_x = np.linspace(restriction[0][0], restriction[1][0], 201)
        coord_y = np.linspace(restriction[0][1], restriction[1][1], 201)
        coord_x_grid, coord_y_grid = np.meshgrid(coord_x, coord_y)
        coord_z_grid = np.zeros(coord_x_grid.shape)
        for i in range(coord_x_grid.shape[0]):
            for j in range(coord_x_grid.shape[1]):
                coord_z_grid[i][j] = f([coord_x_grid[i][j], coord_y_grid[i][j]])

        plt.figure("optimization_steps", figsize=(8, 8))
        cl = plt.contour(coord_x_grid, coord_y_grid, coord_z_grid, cmap='inferno', levels=5)
        plt.clabel(cl, fmt=FuncFormatter(format_cont))

        color_quiver_and_point = 'r'
        plt.plot(x[:k, 0], x[:k, 1], f'o{color_quiver_and_point}', markersize=3.5)
        arrow_x = x[:k, 0]
        arrow_y = x[:k, 1]
        plt.quiver(arrow_x[:-1], arrow_y[:-1], arrow_x[1:] - arrow_x[:-1], arrow_y[1:] - arrow_y[:-1], angles='xy',
                   scale_units="xy", scale=1, color=color_quiver_and_point, width=0.004)
        plt.xlim([restriction[0][0] - 0.5, restriction[1][0] + 0.5])
        plt.ylim([restriction[0][1] - 0.5, restriction[1][1] + 0.5])
        plt.title("Level lines and optimization steps")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.tight_layout()
        plt.grid(which='major')

        plt.show()

    def gradient(f, x, h=1e-4):
        grad = np.zeros(x.size)
        for k in range(x.size):
            dx = np.zeros(x.size)
            dx[k] = h
            grad[k] = (f(x + dx) - f(x - dx)) / (2 * h)
        return grad

    x_0 = np.array(x_0)
    limit = np.array(limit)
    n = 6000
    x = np.zeros((n, x_0.size))
    d = np.zeros((n, x_0.size))
    w = np.zeros(n)

    d_0 = -gradient(func, x_0)

    x[0] = x_0
    d[0] = d_0
    w[0] = np.sqrt(d[0] @ d[0])

    i = 1

    if name_method == "GD":
        beta = lambda j: 0
    elif name_method == "CG_FR":
        beta = lambda j: ((gradient(func, x[j]) @ gradient(func, x[j])) /
                          (gradient(func, x[j - 1]) @ gradient(func, x[j - 1])))
    elif name_method == "CG_PR1":
        beta = lambda j: (gradient(func, x[j]) @ (gradient(func, x[j]) - gradient(func, x[j - 1])) /
                          (gradient(func, x[j]) - gradient(func, x[j - 1]) @ d[j - 1]))
    elif name_method == "CG_PR2":
        beta = lambda j: (-gradient(func, x[j]) @ (gradient(func, x[j]) - gradient(func, x[j - 1])) /
                          (gradient(func, x[j - 1]) @ d[j - 1]))
    else:
        warning = ("Данный метод минимизации ещё неопределён. "
                   "Методы доступные для использования:'GD', 'CG_FR', 'CG_PR1', 'CG_PR2'.")
        raise NameError(warning)

    if name_linear_method == "GS":
        line_search = golden_search
    elif name_linear_method == "BS":
        line_search = binary_search
    elif name_linear_method == "GA":
        line_search = goldstein_armijo
    else:
        warning = ("Данный метод линейного поиска ещё неопределён. "
                   "Методы доступные для использования: 'GS', 'BS', 'GA'.")
        raise NameError(warning)

    while w[i - 1] > eps:
        x[i] = x[i - 1] + line_search(func, d[i - 1], x[i - 1], limit) * d[i - 1]
        d[i] = -gradient(func, x[i]) + beta(i) * d[i - 1]
        w[i] = np.sqrt(d[i] @ d[i])
        i += 1

    np.set_printoptions(precision=3, suppress=True)
    print(f"Результат расчёта:\n"
          f"Вектор обобщённых координат: {x[i - 1]};\n"
          f"Значение целевой функции: {func(x[i - 1]):.3f};\n"
          f"Количество итераций: {i - 1}.")

    contour_plot(func, limit, x, i)

    return x[i - 1]