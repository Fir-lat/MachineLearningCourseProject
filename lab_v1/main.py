import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from yabox.problems import Levy


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def f(x):
    result = []
    for i in range(len(x)):
        result.append(np.random.randint(100))
    return result


x = np.linspace(0, 10, 500)
y = np.ceil(x) + np.random.normal(0, 1, 500)
z = f(x)
plt.scatter(x, z, s=1)
plt.show()

# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
