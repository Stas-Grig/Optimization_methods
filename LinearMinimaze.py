import numpy as np
import matplotlib.pyplot as plt


def __find_max_step(vec_0, direction, limit):
    """ Данная функция позволяет найти максимально возможный шаг для данного направления движения линейной минимизациии;
    Это максимальное значение шага, при котором не случится выхода за ограничения"""
    for i in range(2):
        for j in range(limit[i].size):
            if direction[j] != 0:
                cur_step = (limit[i][j] - vec_0[j]) / direction[j]
                cur_vec = vec_0 + cur_step * direction
                restriction = np.zeros((2, limit[i].size - 1))
                for k in range(2):
                    restriction[k] = np.concatenate((limit[k, :j], limit[k, j + 1:]))
                flag = True
                x_tr = np.concatenate((cur_vec[:j], cur_vec[j + 1:]))
                for g in range(len(x_tr)):
                    if not (restriction[0, g] <= x_tr[g] <= restriction[1, g]):
                        flag = False
                if flag and cur_step > 0:
                    return cur_step


def binary_search(func, direction, x_0, limit):
    def f_x(x):
        return func(x_0 + x * direction)

    a = 0
    b = __find_max_step(x_0, direction, limit)

    """ 
    # Этот участок кода необходим для построения графика функции в данном направлении
    
    x_plot = np.arange(a, b, (b - a) / 1000)
    f_plot = np.zeros(x_plot.size)
    for i in range(x_plot.size):
        f_plot[i] = f_x(x_plot[i])
    """
    eps = b / 1e8

    delta = eps / 2

    while b - a > eps:
        x_left = (a + b - delta) / 2
        x_right = (a + b + delta) / 2
        if f_x(x_left) < f_x(x_right):
            b = x_right
        else:
            a = x_left
    """
    # Cамо построение графика 
    plt.plot(x_plot, f_plot)
    plt.plot((b + a) / 2, f_x((b + a) / 2), "*", markersize=10)
    plt.grid(which="major")
    plt.show()
    """
    return (b + a) / 2


def golden_search(func, direction, x_0, limit):
    def f_x(x):
        return func(x_0 + x * direction)

    a = 0
    b = __find_max_step(x_0, direction, limit)

    """
    # Этот участок кода необходим для построения графика функции в данном направлении

    x_plot = np.arange(a, b, (b - a) / 100)
    f_plot = np.zeros(x_plot.size)
    for i in range(x_plot.size):
        f_plot[i] = f_x(x_plot[i])
    """
    eps = b / 1e8

    x_left = a + (3 - np.sqrt(5)) / 2 * (b - a)
    x_right = b - (3 - np.sqrt(5)) / 2 * (b - a)

    while b - a > eps:
        if f_x(x_left) < f_x(x_right):
            b = x_right
            x_right = x_left
            x_left = a + (3 - np.sqrt(5)) / 2 * (b - a)
        else:
            a = x_left
            x_left = x_right
            x_right = b - (3 - np.sqrt(5)) / 2 * (b - a)

    """
    # Cамо построение графика 

    plt.plot(x_plot, f_plot)
    plt.plot((b + a) / 2, f_x((b + a) / 2), "*", markersize=10)
    plt.grid(which="major")
    plt.show()
    """
    return (b + a) / 2


def goldstein_armijo(func, direction, x_0, limit, p=0.75):
    def f_x(x):
        return func(x_0 + x * direction)

    def gradient_func(f, x, h=1e-4):
        grad = np.zeros(x.size)
        for k in range(x.size):
            dx = np.zeros(x.size)
            dx[k] = h
            grad[k] = (f(x + dx) - f(x - dx)) / (2 * h)
        return grad

    def fi_1(step):
        return func(x_0) + p * step * gradient_func(func, x_0) @ direction

    def fi_2(step):
        return func(x_0) + (1 - p) * step * gradient_func(func, x_0) @ direction

    max_step = __find_max_step(x_0, direction, limit)

    """
    # Этот участок кода необходим для построения графика функции в данном направлении
    x_plot = np.arange(0, max_step, max_step / 100)
    f_plot = np.zeros(x_plot.size)
    fi_1plot = np.zeros(x_plot.size)
    fi_2plot = np.zeros(x_plot.size)
    for i in range(x_plot.size):
        f_plot[i] = f_x(x_plot[i])
        fi_1plot[i] = fi_1(x_plot[i])
        fi_2plot[i] = fi_2(x_plot[i])
    """
    eps = max_step / 1e8
    result_step = max_step * 10 / 7
    while ((fi_1(result_step) >= f_x(result_step) or fi_2(result_step) <= f_x(result_step) or result_step > max_step)
           and result_step > eps):
        if fi_2(result_step) <= f_x(result_step) or result_step > max_step:
            result_step *= 0.6
        else:
            result_step *= 1.4

    r"""
    # Cамо построение графика 
    plt.figure("GA")
    plt.plot(x_plot, f_plot)
    plt.plot(x_plot, fi_1plot)
    plt.plot(x_plot, fi_2plot)
    plt.legend([r'fi($\alpha$)', r'$fi_1$($\alpha$)', r'$fi_2$($\alpha$)'])
    plt.plot(result_step, f_x(result_step), "*", markersize=10)
    plt.grid(which="major")
    plt.show()
    print(x_0 + result_step * direction)
    print(max_step, result_step)
    """
    return result_step