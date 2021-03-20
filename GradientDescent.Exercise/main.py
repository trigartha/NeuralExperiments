# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

dataframe = pd.read_csv('c:\\Users\\Triga\\Documents\\Kika\\P3eg\\3.Exercise\\test_scores.csv')


def predict_using_sklearn():
    r = LinearRegression()
    r.fit(dataframe[['math']], dataframe.cs)
    return r.coef_, r.intercept_


def gradient_descent(x, y):
    # interval approximating 0
    m_curr = b_curr = 0

    # training
    iterations = 1000

    # amount datapoints (we know they are the same for x,y)
    n = len(x)

    # learning rate
    # learning_rate = 0.001
    # learning_rate = 0.01
    # learning_rate = 0.009
    learning_rate = 0.0002

    # to calculate best cost ?
    cost_previous = 0

    # formulas
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr

        # calculate cost for printing
        # val**2 = val square
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        # m derivative
        md = -(2 / n) * sum(x * (y - y_predicted))

        # b derivative
        bd = -(2 / n) * sum((y - y_predicted))

        # adjust currents
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # stop wehn cost is in range of rel_tol=1e-20
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost

        # print iterations
        print("b {}, m {}, cost {}, iteration {}".format(b_curr, m_curr, cost, i))

        return m_curr, b_curr


# numpy arrays for matrix multiplications
x = np.array(dataframe.math)
y = np.array(dataframe.cs)

# gradient_descent(x, y)

m, b = gradient_descent(x, y)
print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

m_sklearn, b_sklearn = predict_using_sklearn()
print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))
