import numpy as np

# pythran export long_func(int)


def long_func(n):
    a = np.arange(n)
    return a ** 3


# pythran export long_func1(int)


def long_func1(n):
    nloops = 10000
    a = np.arange(n // nloops)
    result = a
    for i in range(nloops):
        result += a ** 3 + a ** 2 + 2
