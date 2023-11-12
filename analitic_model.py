from sympy import N, tan, cos, sin, Symbol
from sympy.plotting import plot
from sympy.physics.units import deg


"""Функция вычисления текущей высоты подъёма (1.1)"""


def height_calc(v0: int, alpha: int, x: float) -> float:
    return x * tan(alpha * deg.scale_factor) - G * (x**2)/(2 * v0**2 * cos(alpha * deg.scale_factor)**2)


"""Функция построения графика траектории тела (1.2)"""


def plot_trajectory(v0: int, alpha: int):
    """Максимальная дальность полёта"""
    L = v0**2*sin(2*alpha * deg.scale_factor)/G
    plot(x * tan(alpha * deg.scale_factor) - G*(x**2)/(2*v0**2*cos(alpha * deg.scale_factor)**2), (x, 0, L))


if __name__ == '__main__':
    G = 9.8
    """ параметры для вычисления текущей высоты подъема """
    alpha = 10
    v0 = 100
    x = 50
    print(N(height_calc(v0, alpha, 50)))

    """ интерактивный график траектории тела """
    x = Symbol('x')
    plot_trajectory(v0, alpha)
