import numpy as np
import matplotlib as mpl
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

#退火方法3D网格中求解数个电荷电势最低分布

import matplotlib.pyplot as plt

# 设置网格分辨率
le=1
#电荷数
print("charges amount?")
amount = int(input())
#多少次输出一次图像
print("interval times?")
n = int(input())
# 生成初坐标
xy = np.column_stack((np.random.uniform(0, le, amount), np.random.uniform(0, le, amount), np.random.uniform(0, le, amount)))

# 计算势能
def solve_p(xy):
    # 构建 k-d 树
    tree = cKDTree(xy)
    # 查询树以找到每个点的最近邻居
    distances, indices = tree.query(xy, k=2)
    # 第一个返回的邻居是点自身，所以我们取第二个
    distances = distances[:, 1]
    # 计算势能并求和
    potentials = np.sum(1.0 / distances)
    return potentials

# 生成新坐标时排除和其他坐标一样的函数
def randintxy_except(xy, i):
    for _ in range(100):
        a = np.random.uniform(0, le)
        b = np.random.uniform(0, le)
        c = np.random.uniform(0, le)
        if all(((xy[ii,0] != a or xy[ii,1] != b or xy[ii,2]!=c) and 0<=a<=le and 0<=b<=le and 0<=c<=le) for ii in range(amount)):
            return np.array([a, b, c])
    return np.array([xy[i,0], xy[i,1], xy[i,2]])

# 模拟退火算法一次
def simulated_annealing(xy,ii):
    current_p = solve_p(xy)
    new_xy = np.copy(xy)
    i = random.randint(0, amount-1)
    new_xy[i,:] = randintxy_except(xy, i )
    new_p = solve_p(new_xy)
    if new_p < current_p:
        xy = new_xy
        current_p = new_p
    else:
        p = random.random()
        if p > np.exp(((current_p - new_p)/current_p*100000/(ii+1))):
            xy = new_xy
            current_p = new_p
    return xy, current_p

#画图函数
def draw_3d(xy,i,t,pmin,last_pmin,p0,le,n,amount,potentials):
    fig = plt.figure(figsize=(10, 5))
    cm = plt.get_cmap("coolwarm")  # 色图

    # First subplot for 3D plot
    col = [cm(float(xy[i,2])/(le)) for i in range(amount)]
    ax = fig.add_subplot(121, projection='3d')  # Modify the subplot number to 121
    # Generate colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=le)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    #draw
    ax.view_init(15, 60, 0)
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, depthshade=True,s=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Iteration: {i+1}   Last {n} times: {t:.2f}s')  # Add title and time

    # Second subplot for potential plot
    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(potentials)), potentials, lw=2)
    ax2.set_xlim(0, len(potentials))
    ax2.set_ylim(min(potentials), max(potentials))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Potential Energy')
    ax2.set_title('Potential Energy vs Iteration')

    plt.tight_layout()
    plt.savefig(f"C:/Users/surface/Onedrive/CloudyLake Programming/Product/charge_anneal.png", dpi=300)
    plt.close(fig)

def main():
    global xy,n
    t1 = time.time()
    last_pmin=0
    potentials = []
    p0=solve_p(xy)
    for i in range(100000):
        xy, pmin = simulated_annealing(xy,i)
        potentials.append(pmin)
        if (i+1)%n==0:
            t2 = time.time()
            t=t2-t1
            t1 = time.time()
            draw_3d(xy,i,t,pmin,last_pmin,p0,le,n,amount,potentials)
            last_pmin=pmin
    print(xy)

# 启动！
main()
