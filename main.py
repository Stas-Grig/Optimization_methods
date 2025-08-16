from GradientOptimization import gradient_methods
from PlotFunc import plot_surface, contour_plot


def f(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x_0 = [5, 5]
limit = [[-5, -5], [5, 5]]

plot_surface(f, limit)
contour_plot(f, limit)

gradient_methods(f, x_0, limit, 'CG_PR2', 'GS')
