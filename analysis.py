import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


""" 2.1 Создаём датафрейм из файла .csv """


df = pd.read_csv('data/dataset.csv',
                 usecols=['velocity', 'a', 'b'])


""" 2.2 Вычисление квантилей для скорости """


vel_quantile = df['velocity'].quantile([0.25, 0.5, 0.75])
print(vel_quantile)


""" 2.3 Параметры линейной регрессии, восстанавливающей 
        зависимость между параметром a и скоростью """
X = df[['a']]
y = df['velocity']

model = LinearRegression()
model.fit(X, y)
y_predict = model.predict(X)

print(f'Коэффициенты b\u2081: {model.coef_[0]}, b\u2080: {model.intercept_}')


""" 2.4 График зависимости между параметром a и скоростью """


plt.scatter(X, y, color='blue')
plt.plot(X, y_predict, color='black')
plt.title('Зависимость коэфициента "a" от скорости ')
plt.xlabel('a')
plt.ylabel('velocity')
plt.show()


""" 2.5 Создайте функцию, которая по заданному значению скорости
    и найденным параметрам регрессии определяет значение
    коэффициента сопротивления a. """


def a_from_vel(b0: float, b1: float, velocity: float) -> float:
    a = (velocity - b0)/b1
    return a


print(a_from_vel(model.intercept_, model.coef_[0], 216))


X_b = df[['b']]

y_b = df['velocity'].apply(lambda x: x**2)

model.fit(X_b, y_b)
y_predict_b = model.predict(X_b)


"""  2.6 Параметры регрессии, отражающей зависимость между
         параметром b и квадратом скорости. """

print(f'Коэффициенты b\u2081: {model.coef_[0]}, b\u2080: {model.intercept_}')


""" 2.7 Постройте график второй зависимости и создайте функцию,
        позволяющую определить значение коэффициента
        сопротивления b по заданному значению скорости. """

plt.scatter(X_b, y_b, color='blue')
plt.plot(X_b, y_predict_b, color='black')
plt.title('Зависимость коэфициента "b" от скорости ')
plt.xlabel('b')
plt.ylabel('velocity')
plt.show()


def b_from_vel(b0: float, b1: float, velocity: float) -> float:
    b = (velocity**2 - b0)/b1
    return b


print(b_from_vel(model.intercept_, model.coef_[0], 216))
