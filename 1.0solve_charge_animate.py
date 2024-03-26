#模拟在100*100网格中电荷运动

import numpy as np
import random
import math

# 设置网格
le = 100
xy = np.array([[0,0],[0,0],[0,0],[0,0]])

# 计算势能
def solve_p(xy):
    pi = np.array([0.0]*4)
    for i1 in range(len(pi)):
        for i2 in range(len(pi)):
            d = np.sqrt((xy[i1][0]-xy[i2][0])**2 + (xy[i1][1]-xy[i2][1])**2)
            if d != 0:
                pi[i1] = pi[i1] + (1.0/d)
    return np.sum(pi)

# 模拟退火算法
def simulated_annealing(xy, T):
    current_p = solve_p(xy)
    for t in range(T, 0, -1):
        new_xy = xy.copy()
        i = random.randint(0, 3)
        new_xy[i] = [random.randint(0, le), random.randint(0, le)]
        new_p = solve_p(new_xy)
        if new_p < current_p or random.random() < math.exp((current_p - new_p) / t):
            xy = new_xy
            current_p = new_p
    return xy, current_p

# 启动！
xy, pmin = simulated_annealing(xy, 10000)
print(xy)
print(pmin)
