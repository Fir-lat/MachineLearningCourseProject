import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from DifferentialEvolution.__init__ import DifferentialEvolution
from simplePLR.__init__ import SimplePLR


class PLR(SimplePLR):

    def __init__(self, x, y):
        super().__init__(x, y)

    def regression_matrix(self, breaks, x):
        return super().regression_matrix(breaks, x)

    def simple_plr_v1(self, breaks):
        return super().simple_plr_v1(breaks)

    def predict(self, x):
        return super().predict(x)

    def calculate_slopes(self):
        super().calculate_slopes()

    def standard_errors(self, method='linear', step_size=1e-4):
        super().standard_errors(method, step_size)

    def r_squared(self):
        return super().r_squared()

    def linear_regression(self, A):
        return super().linear_regression(A)

    def plot(self):
        super().plot()

    def simple_plr(self, mid_breaks):
        mid_breaks = np.sort(mid_breaks)
        breaks = np.zeros(len(mid_breaks) + 2)
        breaks[0], breaks[1:-1], breaks[-1] = self.break_0, mid_breaks, self.break_n
        A = self.regression_matrix(breaks, self.x_data)

        # 求解线性回归
        self.ss_res = self.linear_regression(A)
        return self.ss_res

    def fit(self, n_segments):
        self.segments = int(n_segments)
        self.parameters = self.segments + 1

        # 计算断点的域，为[break_0,break_n]
        bounds = np.zeros((self.segments - 1, 2))
        bounds[:, 0] = self.break_0
        bounds[:, 1] = self.break_n

        # 差分进化算法
        result = DifferentialEvolution.differential_evolution(self.simple_plr, bounds)
        self.ss_res = result[1]
        mid_breaks = np.sort(result[0])
        self.breakpoints = np.zeros(len(mid_breaks) + 2)
        self.breakpoints[0], self.breakpoints[1:-1], self.breakpoints[-1] = self.break_0, mid_breaks, self.break_n

        #
        self.simple_plr_v1(self.breakpoints)

        return self.breakpoints

    def fit_debug(self, n_segments):
        self.segments = int(n_segments)
        self.parameters = self.segments + 1

        # 计算断点的域，为[break_0,break_n]
        bounds =    np.zeros((self.segments - 1, 2))
        bounds[:, 0] = self.break_0
        bounds[:, 1] = self.break_n

        # 差分进化算法
        result_list = list(DifferentialEvolution.differential_evolution_debug
                           (self.simple_plr, bounds, iterations=3000))
        x, f = zip(*result_list)
        plt.plot(f, label='segments={}'.format(n_segments))

        self.ss_res = result_list[-1][1]
        mid_breaks = np.sort(result_list[-1][0])
        self.breakpoints = np.zeros(len(mid_breaks) + 2)
        self.breakpoints[0], self.breakpoints[1:-1], self.breakpoints[-1] = self.break_0, mid_breaks, self.break_n

        #
        self.simple_plr_v1(self.breakpoints)

        return self.breakpoints

