import matplotlib.pyplot as plt

import math

import numpy as np

import pandas as pd

from sympy import N, sin, cos, sqrt
from sympy.physics.units import deg

from scipy.integrate import odeint

from sklearn.linear_model import LinearRegression



G = 9.8
v0 = 300

a = 0.07
b = 0
alpha = 30

""" 3.1 Создайте функцию, которая вычисляет правую часть системы обыкновенных дифференциальных уравнений. """

def odu(s, tau, a, b, alpha):
    dVxdt = -a * sin(alpha * deg.scale_factor) * s[0] - b * sin(alpha * deg.scale_factor) * sqrt(s[0]**2 + s[1]**2) * s[0]
    dVydt = -sin(alpha * deg.scale_factor) - a * sin(alpha * deg.scale_factor) * s[1] - b * sin(alpha * deg.scale_factor) * sqrt(s[0]**2 + s[1]**2) * s[1]
    dXdt = s[0]/(2 * cos(alpha * deg.scale_factor))
    dYdt = 2 * s[1]/(sin(alpha * deg.scale_factor))
    return [dVxdt, dVydt, dXdt, dYdt]


time = []
x_values = []
y_values = []
vx_values = []
vy_values = []
Vx = cos(alpha * deg.scale_factor)
Vy = sin(alpha * deg.scale_factor)
X = 0
Y = 0
T = v0 * N(sin(alpha * deg.scale_factor))/G
t = 0
dt = 1 / T

""" Метод Рунге-Кутта 4-го порядка """
while t < T:

    dVx1 = (-a * sin(alpha * deg.scale_factor) * Vx - b * sin(alpha * deg.scale_factor) * sqrt(Vx**2 + Vy**2) * Vx) * dt
    dVy1 = (-sin(alpha * deg.scale_factor) - a * sin(alpha * deg.scale_factor) * Vy - b * sin(alpha * deg.scale_factor) * sqrt(Vx**2 + Vy**2) * Vy) * dt
    dX1 = Vx/(2 * cos(alpha * deg.scale_factor))
    dY1 = 2 * Vy/(cos(alpha * deg.scale_factor))
    dVx2 = (-a * sin(alpha * deg.scale_factor) * (Vx + dVx1/2) - b * sin(alpha * deg.scale_factor) * sqrt((Vx + dVx1/2)**2 + (Vy + dVy1/2)**2) * (Vx + dVx1/2)) * dt
    dVy2 = (-sin(alpha * deg.scale_factor) - a * sin(alpha * deg.scale_factor) * (Vy + dVy1/2) - b * sin(alpha * deg.scale_factor) * sqrt((Vx + dVx1/2)**2 + (Vy + dVy1/2)**2) * (Vy + dVy1/2)) * dt
    dX2 = (Vx + dVx1/2)/(2 * cos(alpha * deg.scale_factor))
    dY2 = 2 * (Vy + dVy1/2)/(cos(alpha * deg.scale_factor))
    dVx3 = (-a * sin(alpha * deg.scale_factor) * (Vx + dVx2/2) - b * sin(alpha * deg.scale_factor) * sqrt((Vx + dVx2/2)**2 + (Vy + dVy2/2)**2) * (Vx + dVx2/2)) * dt
    dVy3 = (-sin(alpha * deg.scale_factor) - a * sin(alpha * deg.scale_factor) * (Vy + dVy2/2) - b * sin(alpha * deg.scale_factor) * sqrt((Vx + dVx2/2)**2 + (Vy + dVy2/2)**2) * (Vy + dVy2/2)) * dt
    dX3 = (Vx + dVx2/2)/(2 * cos(alpha * deg.scale_factor))
    dY3 = 2 * (Vy + dVy2/2)/(cos(alpha * deg.scale_factor))
    dVx4 = (-a * sin(alpha * deg.scale_factor) * (Vx + dVx3) - b * sin(alpha * deg.scale_factor) * sqrt((Vx + dVx3)**2 + (Vy + dVy3)**2) * (Vx + dVx3)) * dt
    dVy4 = (-sin(alpha * deg.scale_factor) - a * sin(alpha * deg.scale_factor) * (Vy + dVy3) - b * sin(alpha * deg.scale_factor) * sqrt((Vx + dVx3)**2 + (Vy + dVy3)**2) * (Vy + dVy3)) * dt
    dX4 = (Vx + dVx3)/(2 * cos(alpha * deg.scale_factor))
    dY4 = 2 * (Vy + dVy3)/(cos(alpha * deg.scale_factor))
    Vx += (dVx1 + 2*dVx2 + 2*dVx3 + dVx4) / 6
    Vy += (dVy1 + 2*dVy2 + 2*dVy3 + dVy4) / 6
    X += (dX1 + 2*dX2 + 2*dX3 + dX4) / 6
    Y += (dY1 + 2*dY2 + 2*dY3 + dY4) / 6
    vx_values.append(Vx)
    vy_values.append(Vy)
    x_values.append(X)
    y_values.append(Y)
    time.append(t)
    t += dt

# print(vx_values)
# print(vy_values)
print(x_values)
print(y_values)
""" 3.3 Найдите решение системы ОДУ в случае отсутствия сопротивления, то есть
    при параметрах a и b, равных нулю. Используйте значение угла — 30°. """


a = 0
b = 0
alpha = 30
T = float(v0 * N(sin(alpha * deg.scale_factor))/G)
t = np.linspace(0, T)
tau = 2 * t/T

""" Нулевые значения """
s0 = [N(cos(alpha * deg.scale_factor)), N(sin(alpha * deg.scale_factor)), 0, 0]
s = odeint(odu, s0, tau, args=(a, b, alpha))

plt.plot(tau, s[:,0], label="Vx(t)")
plt.plot(tau, s[:,1], label="Vy(t)")
plt.plot(tau, s[:,2], label="X(t)")
plt.plot(tau, s[:,3], label="Y(t)")
plt.xlabel('tau')
plt.ylabel('Vx, Vy, X, Y')
plt.grid()
plt.legend()
plt.show()


""" 3.4 Найдите решение системы ОДУ, полагая, что значение скорости равно третьему квартилю выборки, 
    и соответствующим образом вычисляя параметры a и b с помощью функций, реализованных ранее. """

def a_from_vel(b0: float, b1: float, velocity: float) -> float:
    a = (velocity - b0)/b1
    return a

def b_from_vel(b0: float, b1: float, velocity: float) -> float:
    b = (velocity**2 - b0)/b1
    return b

df = pd.read_csv('data/dataset.csv',
                 usecols=['velocity', 'a', 'b'])

v0 = np.percentile(df.velocity, 75)
X = df[['a']]
y = df['velocity']
X_b = df[['b']]
y_b = df['velocity'].apply(lambda x: x**2)

model = LinearRegression()
model.fit(X, y)
y_predict = model.predict(X)

model.fit(X_b, y_b)
y_predict_b = model.predict(X_b)


a = a_from_vel(model.intercept_, model.coef_[0], v0)
b = b_from_vel(model.intercept_, model.coef_[0], v0)

s0 = [N(cos(alpha * deg.scale_factor)), N(sin(alpha * deg.scale_factor)), 0, 0]
s = odeint(odu, s0, tau, args=(a, b, alpha))


""" 3.5 В обоих случаях постройте графики траектории в осях, 
        соответствующих безразмерным переменным. """


plt.plot(tau, s[:, 0], label="Vx(t)")
plt.plot(tau, s[:, 1], label="Vy(t)")
plt.plot(tau, s[:, 2], label="X(t)")
plt.plot(tau, s[:, 3], label="Y(t)")
plt.xlabel('tau')
plt.ylabel('Vx, Vy, X, Y')
plt.grid()
plt.legend()
plt.show()


L = v0 ** 2 * N(sin(2 * alpha * deg.scale_factor)) / G
H = (v0 ** 2 * N(sin(alpha * deg.scale_factor)) ** 2) / (2 * G)
T = 2 * v0 * N(sin(alpha * deg.scale_factor))
Tpod = v0 * N(sin(alpha * deg.scale_factor))

print('Дальность полёта (м).', L)
print('Максимальную высоту подъёма (м).', H)
print('Время полёта (сек.).', T)
print('Время, необходимое для подъёма на максимальную высоту (сек.).', Tpod)
print('Модуль скорости в момент падения (м/с).', math.sqrt(Vx**2 + Vy**2))