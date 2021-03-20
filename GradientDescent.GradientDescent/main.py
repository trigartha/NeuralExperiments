# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np


# find m and b


def gradient_descent(x, y):
    # gradient interval to 0
    m_curr = b_curr = 0

    # amount of training
    # when to stop or increase iterations?
    # start with a small amount and check if learning rate decreases
    iterations = 10000

    # n is the length of the data points - WE ARE ASSUMING length of X and Y is the same
    # (if this is not the case we need 2 lines here)
    n = len(x)

    # learning rate - some value close to 0 - this might be trial and error
    # learning_rate = 0.0001
    # cost was increasing at 0.0001 - this is wrong, cost needs to decrease
    # learning_rate = 0.001
    # test with bigger learning rate - increases so wrong
    # learning_rate = 0.1
    # decreases at 0.01 and gets closer to optimun values m (2) and b (3)
    # learning_rate = 0.01
    # learning_rate = 0.09 creates nans
    # best learning rate = 0.08 (closest to 0.1 with decreasing cost and values b & m reaching optimum values)
    # cost stays almost the same at this point - sweet spot for algorithm ~ floating point comparison
    learning_rate = 0.08

    # use slope formula
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr

        # print cost to check if program runs well
        cost = (1/n)*sum([val **2 for val in (y-y_predicted)])

        # m derivative
        md = -(2/n)*sum(x*(y-y_predicted))

        # b derivative
        bd = -(2/n)*sum(y-y_predicted)

        # adjust m current
        m_curr = m_curr - learning_rate * md

        # adjust b current
        b_curr = b_curr - learning_rate * bd

        # print values at each iteration
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))


# use numpy array because matrix multiplication is more convenient this way
# and it's somehow faster than regular python arrays

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
