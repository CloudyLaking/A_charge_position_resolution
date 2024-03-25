#退火方法网格中求解数个电荷电势最低分布

import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

# 设置网格和电荷数
print("dpi=?*?")
le = int(input())
print("charge amount?")
amount = int(input())
print("times?")
n = int(input())
xy = np.column_stack((np.random.randint(0, le, amount), np.random.randint(0, le, amount)))

# 计算势能
def solve_p(xy):
    pi = np.zeros(amount)
    for i1 in range(amount):
        for i2 in range(amount):
            d = np.sqrt((xy[i1,0]-xy[i2,0])**2 + (xy[i1,1]-xy[i2,1])**2)
            if d != 0:
                pi[i1] += 1.0/d
    return np.sum(pi)

# 生成新坐标时排除和其他坐标一样的函数
def randintxy_except(xy, i):
    for _ in range(100):
        a = random.randint(xy[i,0]-le//20-1, xy[i,0]+le//20+1); b = random.randint(xy[i,1]-le//20-1, xy[i,1]+le//20+1)
        if all(((xy[n,0] != a or xy[n,1] != b) and 0<=a<=le and 0<=b<=le) for n in range(amount)):
            return np.array([a, b])
    return np.array([xy[i,0], xy[i,1]])

# 模拟退火算法
def simulated_annealing(xy):
    current_p = solve_p(xy)
    new_xy = np.copy(xy)
    i = random.randint(0, amount-1)
    new_xy[i,:] = randintxy_except(xy, i )
    new_p = solve_p(new_xy)
    if new_p < current_p:
        xy = new_xy
        current_p = new_p
    return xy, current_p

def main():
    global xy
    t1 = datetime.datetime.now()
    for i in range(n):
        xy, pmin = simulated_annealing(xy)
    print(xy)
    print(pmin)
    t2 = datetime.datetime.now()
    print(t2 - t1,"s")
    plt.scatter(xy[:, 0], xy[:, 1], color='pink')
    plt.axhline(0, color='grey', linewidth=1, linestyle=':')
    plt.axhline(le, color='grey', linewidth=1, linestyle=':')
    plt.axvline(0, color='grey', linewidth=1, linestyle=':')
    plt.axvline(le, color='grey', linewidth=1, linestyle=':')
    plt.xlim(-10, le + 10)
    plt.ylim(-10, le + 10)
    plt.title(f"location of charge, t={i+1}")
    plt.show()
    plt.savefig(f"C:\\Users\\13080\\Desktop\\Onedrive\\CloudyLake Programming\\Product\\charge.png")

# 启动！
main()
