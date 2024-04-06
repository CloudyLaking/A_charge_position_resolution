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
    return xy, current_p

#画图函数
def draw_3d(xy,i,t,pmin,last_pmin,p0,le,n,amount,potentials):
    fig = plt.figure(figsize=(10, 10))
    cm = plt.get_cmap("coolwarm")  # 色图

    # First subplot for 3D plot
    col = [cm(float(xy[i,2])/(le)) for i in range(amount)]
    ax1 = fig.add_subplot(221, projection='3d')  # Modify the subplot number to 221
    # Generate colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=le)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5)
    #draw
    ax1.view_init(15, 60, 0)
    ax1.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, depthshade=True,s=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Iteration with anneal: {i+1} \n  Last {n} times: {t:.2f}s \n  Charges: {amount}')  # Add title, time, and charges

    # Second subplot for potential plot
    ax2 = fig.add_subplot(222)
    ax2.plot(range(len(potentials)), potentials, lw=2)
    ax2.set_xlim(0, len(potentials))
    ax2.set_ylim(min(potentials), max(potentials))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Potential Energy')
    ax2.set_title('Potential Energy vs Iteration')
    ax2.text(0.5, 0.9, f'Decrease: {last_pmin - pmin}', transform=ax2.transAxes, ha='center')  # Add text for decrease in potential energy

    # Third subplot for 2D plot
    ax3 = fig.add_subplot(223, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = 0.05
    delta = 0.05  # Adjust this value as needed
    z_indices = np.where((xy[:, 2] >= z_values - delta) & (xy[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xy[z_indices, 0]
    y_values = xy[z_indices, 1]

    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        x_min = np.min(x_values)
        x_max = np.max(x_values)
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        ax3.scatter(x_values, y_values, c='orange',s=2)  # Convert col to a numpy array
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title(f'2D Slices at Z = {z_values} \n  Points: {len(x_values)}')  # Add title with number of points
        ax3.set_xlim(0, le)  # Set x-axis limits to 0 and le
        ax3.set_ylim(0, le)  # Set y-axis limits to 0 and le
        ax3.grid(True)

    # Fourth subplot for 2D plot
    ax4 = fig.add_subplot(224, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = 0.5
    delta = 0.05  # Adjust this value as needed
    z_indices = np.where((xy[:, 2] >= z_values - delta) & (xy[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xy[z_indices, 0]
    y_values = xy[z_indices, 1]

    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        x_min = np.min(x_values)
        x_max = np.max(x_values)
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        ax4.scatter(x_values, y_values, c='orange',s=2)  # Convert col to a numpy array
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title(f'2D Slices at Z = {z_values} \n  Points: {len(x_values)}')  # Add title with number of points
        ax4.set_xlim(0, le)  # Set x-axis limits to 0 and le
        ax4.set_ylim(0, le)  # Set y-axis limits to 0 and le
        ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f"C:/Users/surface/Onedrive/CloudyLake Programming/Product/charge_non_anneal.png", dpi=300)
    plt.close(fig)

def main():
    global xy,n
    t1 = time.time()
    last_pmin=0
    potentials = []
    p0=solve_p(xy)
    for i in range(1000000):
        xy, pmin = simulated_annealing(xy,i)
        potentials.append(pmin)
        if (i+1)%n==0:
            t2 = time.time()
            t=t2-t1
            t1 = time.time()
            draw_3d(xy,i,t,pmin,last_pmin,p0,le,n,amount,potentials)
            last_pmin=pmin
    print(xy)
    print(pmin)

# 启动！
main()
