import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


class SimplePLR(object):

    def __init__(self, x, y):
        # 首先需要判断数据是否为ndarray对象
        if isinstance(x, np.ndarray) is False:
            self.x_data = np.array(x)
        else:
            self.x_data = x
        if isinstance(x, np.ndarray) is False:
            self.y_data = np.array(y)
        else:
            self.y_data = y
        self.n_data = self.x_data.size  # 数据集大小
        # 前后两断点就是端点
        self.break_0 = np.min(self.x_data)
        self.break_n = np.max(self.x_data)
        self.breakpoints = None  # 断点
        self.segments = None  # 分段数
        self.parameters = None  # 回归矩阵列数
        self.beta = None  # 线性回归解
        self.ss_res = None  # 残差平方和
        self.slopes = None  # 各分段的斜率
        self.intercepts = None  # 各分段的截距
        self.se = None  # 标准差

    def regression_matrix(self, breaks, x):
        if isinstance(breaks, np.ndarray) is False:
            breaks = np.array(breaks)
        self.breakpoints = np.sort(breaks)
        self.segments = len(self.breakpoints) - 1

        # 从第一列开始生成回归矩阵A
        A_0 = [np.ones_like(x), x - self.breakpoints[0]]
        for i in range(self.segments - 1):
            A_0.append(np.where(x >= self.breakpoints[i + 1], x - self.breakpoints[i+1], 0.0))
        A = np.vstack(A_0).T  # 组装成回归矩阵
        self.parameters = A.shape[1]
        return A

    def simple_plr_v1(self, breaks):
        if isinstance(breaks, np.ndarray) is False:
            breaks = np.array(breaks)
        self.breakpoints = np.sort(breaks)
        A = self.regression_matrix(self.breakpoints, self.x_data)

        # 求解线性回归
        self.ss_res = self.linear_regression(A)
        return self.ss_res

    def predict(self, x):
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        A = self.regression_matrix(self.breakpoints, x)

        # 模型预测
        y_hat = np.dot(A, self.beta)
        return y_hat

    def calculate_slopes(self):
        y_hat = self.predict(self.breakpoints)  # 因为断点不一定是数据点
        # 断点的因变量差值除以断点间距离
        self.slopes = np.divide(
            (y_hat[1:self.segments + 1] - y_hat[0:self.segments]),
            (self.breakpoints[1:self.segments + 1] - self.breakpoints[0:self.segments]))
        self.intercepts = y_hat[0:-1] - self.slopes * self.breakpoints[0:-1]
        return self.slopes

    def standard_errors(self):
        n_b = self.beta.size
        n = self.n_data
        A = self.regression_matrix(self.breakpoints, self.x_data)
        y_hat = np.dot(A, self.beta)
        e = y_hat - self.y_data

        # 残差平方无偏估计，计算标准差
        sigma2 = np.dot(e, e) / (n - n_b)
        A_inv = np.abs(linalg.inv(np.dot(A.T, A)).diagonal())
        self.se = np.sqrt(sigma2 * A_inv)
        return self.se

    def r_squared(self):
        ss_res = self.simple_plr_v1(self.breakpoints)
        y_bar = np.ones(self.n_data) * np.mean(self.y_data)
        y_difference = self.y_data - y_bar
        try:
            ss_tot = np.dot(y_difference, y_difference)
            rsq = 1.0 - (ss_res / ss_tot)
            return rsq
        except linalg.LinAlgError:
            raise linalg.LinAlgError('Singular matrix')

    def linear_regression(self, A):
        try:
            self.beta, self.ss_res, _, _ = linalg.lstsq(A, self.y_data, rcond=None)
        except linalg.LinAlgError:
            raise linalg.LinAlgError('Singular Matrix!')
            self.ss_res = np.inf

        # 当数据集大小不大于回归列数，残差平方和不会被计算，因此需要手动计算
        if self.ss_res.size == 0 or self.n_data <= self.parameters:
            y_hat = np.dot(A, self.beta)
            e = y_hat - self.y_data
            self.ss_res = np.dot(e, e)

        self.calculate_slopes()
        return self.ss_res

    def plot(self):
        x_hat = np.linspace(min(self.x_data), max(self.x_data), num = 10000)
        y_hat = self.predict(x_hat)
        plt.figure()
        plt.plot(self.x_data, self.y_data, 'o', label='Original data')
        plt.plot(x_hat, y_hat, label='Piecewise Linear Regression')
        plt.xlabel('data x')
        plt.ylabel('data y')
        plt.title('breakpoints ' + str(len(self.breakpoints)))
        plt.legend()
        plt.show()
