import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from datetime import datetime

# 设置网格和电荷数
print("dpi=?*?")
le = int(input())
print("电荷数量？")
amount = int(input())
print("模拟次数？")
n = int(input())
xy = np.random.randint(0, le+2, (amount, 2))

# 计算势能
def solve_p(xy):
    pi = np.zeros(amount)
    for i1 in range(amount):
        for i2 in range(amount):
            d = np.sqrt((xy[i1,0]-xy[i2,0])**2 + (xy[i1,1]-xy[i2,1])**2)
            if d != 0:
                pi[i1] += 1.0/d
    return sum(pi)

# 生成新坐标时排除和其他坐标一样的函数
@staticmethod
def randintxy_except(x, y, xy):
    for _ in range(100):
        a = random.randint(x, y)
        b = random.randint(x, y)
        if all(xy[:,0] != a) and all(xy[:,1] != b):
            return [a, b]
    return [0, 0]

# 模拟退火算法
def simulated_annealing(xy):
    current_p = solve_p(xy)
    new_xy = np.copy(xy)
    i = random.randint(1, amount-1)
    new_xy[i,:] = randintxy_except(0, le, xy)
    new_p = solve_p(new_xy)
    if new_p < current_p:
        xy = new_xy
        current_p = new_p
    return xy, current_p

class Simulation:
    def __init__(self, xy, ax):
        self.xy = xy
        self.ax = ax

    def update(self, num):
        self.ax.clear()
        for i, (x, y) in enumerate(self.xy):
            self.ax.scatter(x, y, s=100, c='red')
            self.ax.text(x, y, f'{i}', fontsize=10, ha='right', va='top')
        self.ax.set_xlim(min(self.xy[:, 0])-1, max(self.xy[:, 0])+1)  # Adjust the x-axis to fit the data
        self.ax.set_ylim(min(self.xy[:, 1])-1, max(self.xy[:, 1])+1)  # Adjust the y-axis to fit the data

# 主函数
def main():
    global xy
    global pmin
    t1 = datetime.now()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    pmin = float('inf')
    sim = Simulation(xy, ax)
    for i in range(n):
        xy, pmin = simulated_annealing(xy)
        if i % 100 == 0:
            sim.update(i)  # 更新动画的帧
    anim = FuncAnimation(fig, sim.update, frames=range(0, n, 100), repeat=False)
    anim.save("charge_animation.gif", writer='pillow', fps=3)
    t2 = datetime.now()
    print(xy)
    print(pmin)
    print(t2-t1)

# 发车！
main()

