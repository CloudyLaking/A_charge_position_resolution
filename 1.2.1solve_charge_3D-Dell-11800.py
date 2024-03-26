#退火方法3D网格中求解数个电荷电势最低分布

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# 设置网格分辨率
print("dpi=?*?")
le = int(input())
#电荷数
print("charges amount?")
amount = int(input())
#多少次输出一次图像
print("interval times?")
n = int(input())
# 生成初坐标
xy = np.column_stack((np.random.randint(0, le, amount), np.random.randint(0, le, amount), np.random.randint(0, le, amount)))

# 计算势能
def solve_p(xy):
    pi = np.zeros(amount)
    for i1 in range(amount):
        for i2 in range(amount):
            d = np.sqrt((xy[i1,0]-xy[i2,0])**2 + (xy[i1,1]-xy[i2,1])**2+ (xy[i1,2]-xy[i2,2])**2)
            if d != 0:
                pi[i1] += 1.0/d
    return np.sum(pi)

# 生成新坐标时排除和其他坐标一样的函数
def randintxy_except(xy, i):
    for _ in range(100):
        a = random.randint(xy[i,0]-le//20-1, xy[i,0]+le//20+1)
        b = random.randint(xy[i,1]-le//20-1, xy[i,1]+le//20+1)
        c = random.randint(xy[i,1]-le//20-1, xy[i,1]+le//20+1)
        if all(((xy[ii,0] != a or xy[ii,1] != b or xy[ii,2]!=c) and 0<=a<=le and 0<=b<=le and 0<=c<=le) for ii in range(amount)):
            return np.array([a, b, c])
    return np.array([xy[i,0], xy[i,1], xy[i,2]])

# 模拟退火算法一次
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

#画图函数
def draw_3d(xy,i,t,pmin,le):
    fig = plt.figure(figsize=(6, 6))
    cm = plt.get_cmap("coolwarm")  # 色图
    # 画数据
    col = [cm(float(xy[i,2])/(le)) for i in range(amount)]
    ax = fig.add_subplot(111, projection='3d')
    # Generate colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=le)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

    # 画图
    ax.view_init(40, 60, 0)
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('')  # 隐藏z轴标题
    ax.set_zticks([])  # 隐藏z轴标签
    ax.set_title(f"location of charge, t={i+1} \n last loop:{t}s \n potential:{pmin}")
    plt.savefig("C:/Users/admin/Onedrive/CloudyLake Programming/Product/charge.png", dpi=300)
    plt.close(fig)

#主函数
def main():
    global xy
    for i in range(100000000000000000000000000000):
        t1 = time.time()
        xy, pmin = simulated_annealing(xy)
        if (i+1)%n==0:
            t2 = time.time()
            t=t2-t1
            t1 = time.time()
            draw_3d(xy,i,t,pmin,le)
# 启动！
main()
