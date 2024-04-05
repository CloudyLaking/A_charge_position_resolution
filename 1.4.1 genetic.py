import numpy as np
import matplotlib as mpl
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# 遗传算法中求解数个电荷电势最低分布


# 设置网格分辨率
le = 1
# 电荷数
print("charges amount?")
amount = int(input())
# 多少次输出一次图像
print("interval times?")
n = int(input())
#种群大小
print("population size?")
population= int(input())

# 生成含population个初坐标组的数组
xyxy = np.array([np.column_stack((np.random.uniform(0, le, amount), np.random.uniform(0, le, amount), np.random.uniform(0, le, amount))) for _ in range(100)])


# 计算势能函数
def solve_p(xy):
    # 构建 k-d 树
    tree = cKDTree(xy)
    # 查询树以找到每个点的最近邻居
    distances, indices = tree.query(xy, k=2)
    # 第一个返回的邻居是点自身，所以我们取第二个
    distances = distances[:, 1]
    # 计算势能并求和
    potential = np.sum(1.0 / distances)
    return potential

# 选择，并且返回这一大组里面最好的一组坐标的函数
def select(xyxy):
    potentials = np.array([solve_p(xy) for xy in xyxy])

    probabilities = 1 / potentials
    probabilities /= np.sum(probabilities)
    selected_indices = np.random.choice(len(xyxy), size=len(xyxy)//2, replace=False, p=probabilities)
    selected_xyxy = np.array([xyxy[i] for i in selected_indices])
    
    # Generate new xyxy based on selected xyxy as seed
    new_xyxy = np.array([np.column_stack((np.random.uniform(0, le, amount), np.random.uniform(0, le, amount), np.random.uniform(0, le, amount))) for _ in range(len(selected_xyxy))])
    
    # Combine selected xyxy and new xyxy
    combined_xyxy = np.concatenate((selected_xyxy, new_xyxy))
    
    return combined_xyxy, np.average(potentials)

# 交叉操作函数
def crossover(xy1, xy2):
    i = random.randint(0, len(xy1)-1)
    new_xy1 = np.concatenate((xy1[:i], xy2[i:]))
    new_xy2 = np.concatenate((xy2[:i], xy1[i:]))
    return new_xy1, new_xy2

# 变异操作函数
def mutation(xy):
    i = random.randint(0, len(xy)-1)
    xy[i] = np.array(np.random.uniform(0, le, 3))
    return xy

# 遗传算法

# 画图函数
def draw_3d(xy, i, t, pmin, last_pmin, p0, le, n, amount, potentials):
    fig = plt.figure(figsize=(10, 10))
    cm = plt.get_cmap("coolwarm")  # 色图

    # First subplot for 3D plot
    col = [cm(float(xy[i, 2]) / (le)) for i in range(amount)]
    ax1 = fig.add_subplot(221, projection='3d')  # Modify the subplot number to 223
    # Generate colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=le)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5)
    # draw
    ax1.view_init(15, 60, 0)
    ax1.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, depthshade=True, s=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Iteration with genetic algorithm: {i + 1} \n  Last {n} times: {t:.2f}s \n  Charges: {amount}')  # Add title, time, and charges

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
    plt.savefig(f"C:/Users/surface/Onedrive/CloudyLake Programming/Product/charge_genetic.png", dpi=300)
    plt.close(fig)

def main():
    global xyxy, n
    t1 = time.time()
    last_pmin = 0
    potentials = np.array([])
    p0 = solve_p(xyxy[0])
    for i in range(100000):
        xyxy,pmin= select(xyxy)
        for i1 in range(0, len(xyxy)-1, 2):
            xyxy[i1],xyxy[i1+1] = crossover(xyxy[i1], xyxy[i1+1])
        for i2 in range(len(xyxy)):
            if random.random() < 0.1:  # 10%的变异率   
                xyxy[i2] = mutation(xyxy[i2]) 
        potentials = np.append(potentials, pmin)
        if (i + 1) % n == 0:
            t2 = time.time()
            t = t2 - t1
            t1 = time.time()
            draw_3d(xyxy[0], i, t, pmin, last_pmin, p0, le, n, amount, potentials)
            last_pmin = pmin
    print(xyxy[0])
    print(pmin)

# 启动！
main()