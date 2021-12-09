import numpy as np


class DifferentialEvolution:

    @staticmethod
    def differential_evolution(function, bounds, mutation=0.8, crossover_p=0.7, population_size=20, iterations=1000):
        # 初始化
        dimensions = len(bounds)
        population_regular = np.random.rand(population_size, dimensions)  # 随机生成种群的正则化个体
        min_b, max_b = np.asarray(bounds).T  # 取域边界
        diff = np.fabs(max_b - min_b)  # 样本极差，域长
        populations = min_b + population_regular * diff  # 生成种群
        evaluation = np.asarray([function(individual) for individual in populations])  # 用待优化函数评估种群
        best_index = np.argmin(evaluation)  # 最好的个体索引
        best = populations[best_index]  # 最好的个体

        # 默认迭代1000次
        for i in range(iterations):
            # 遍历每个个体
            for j in range(population_size):
                indexes = [index for index in range(population_size) if index != j]  # 取除j外的个体
                a, b, c = population_regular[np.random.choice(indexes, 3, replace=False)]  # 随机找三个
                mutant = np.clip(a + mutation * (b - c), 0, 1)  # mutation规约到域范围内
                cross_points = np.random.rand(dimensions) < crossover_p  # crossover随机重组
                # 因为本实验中数据集为一维，所以这里很可能不会发生任何重组，所以需手动添加重组
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                candidate_regular = np.where(cross_points, mutant, population_regular[j])  # 候选人正则化
                candidate = min_b + candidate_regular * diff  # 候选人
                f = function(candidate)  # 评估候选人
                if f < evaluation[j]:  # 如果候选人更优
                    evaluation[j] = f
                    population_regular[j] = candidate_regular
                    if f < evaluation[best_index]:  # 如果候选人优于最优
                        best_index = j
                        best = candidate
        return best, evaluation[best_index]
    
    @staticmethod    
    def differential_evolution_debug(function, bounds, mutation=0.8, crossover_p=0.7,
                                     population_size=20, iterations=1000):
        # 初始化
        dimensions = len(bounds)
        population_regular = np.random.rand(population_size, dimensions)  # 随机生成种群的正则化个体
        min_b, max_b = np.asarray(bounds).T  # 取域边界
        diff = np.fabs(max_b - min_b)  # 样本极差，域长
        populations = min_b + population_regular * diff  # 生成种群
        evaluation = np.asarray([function(individual) for individual in populations])  # 用待优化函数评估种群
        best_index = np.argmin(evaluation)  # 最好的个体索引
        best = populations[best_index]  # 最好的个体

        # 默认迭代1000次
        for i in range(iterations):
            # 遍历每个个体
            for j in range(population_size):
                indexes = [index for index in range(population_size) if index != j]  # 取除j外的个体
                a, b, c = population_regular[np.random.choice(indexes, 3, replace=False)]  # 随机找三个
                mutant = np.clip(a + mutation * (b - c), 0, 1)  # mutation规约到域范围内
                cross_points = np.random.rand(dimensions) < crossover_p  # crossover随机重组
                # 因为本实验中数据集为一维，所以这里很可能不会发生任何重组，所以需手动添加重组
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                candidate_regular = np.where(cross_points, mutant, population_regular[j])  # 候选人正则化
                candidate = min_b + candidate_regular * diff  # 候选人
                f = function(candidate)  # 评估候选人
                if f < evaluation[j]:  # 如果候选人更优
                    evaluation[j] = f
                    population_regular[j] = candidate_regular
                    if f < evaluation[best_index]:  # 如果候选人优于最优
                        best_index = j
                        best = candidate
            yield best, evaluation[best_index]  # 迭代器模式，便于后续观察迭代次数对算法收敛的影响


