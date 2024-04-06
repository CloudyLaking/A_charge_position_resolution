import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

#用逐个生成的方法模拟放置电荷寻求最低分布

# 设置网格分辨率
le=int(input("resolution?"))
#电荷数
amount = int(input("charges amount?"))
#多少次输出一次图像
n=int(input("interval times?"))
# 生成初坐标
xyxy = np.array([[0,0,0]])

# 计算势能函数
def solve_p(xyxy):
    # 计算点对之间的距离
    distances = np.linalg.norm(xyxy[:, np.newaxis] - xyxy, axis=2)
    # 将对角线上的元素设置为无穷大，以排除自距离
    np.fill_diagonal(distances, np.inf)
    # 计算倒数之和
    return np.sum(1.0 / distances)


# 严格生成取舍算法一次的函数
def generate(xyxy):
    xyxy0 = np.copy(xyxy)  # 存档初位置
    current_p = float('inf')
    for x in range(le + 1):
        for y in range(le + 1):
            for z in range(le + 1):
                if all(((xyxy[ii, 0] != x or xyxy[ii, 1] != y or xyxy[ii, 2] != z)) for ii in range(len(xyxy))):
                    xy = np.array([x, y, z])
                    xyxy = np.vstack((xyxy0, xy))
                    new_p = solve_p(xyxy)
                    if new_p < current_p:
                        xyxy1 = np.copy(xyxy)
                        current_p = new_p
    p_average = current_p/ np.square(len(xyxy0))
    return xyxy1, p_average

#画图函数
def draw_3d(xy,i,t,pmin,last_pmin,p0,le,n,amount,potentials):
    fig = plt.figure(figsize=(10, 10))
    cm = plt.get_cmap("coolwarm")  # 色图

    # First subplot for 3D plot
    col = [cm(float(xy[i1,2])/(le)) for i1 in range(len(xy))]
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
    ax1.set_title(f'Charges: {i+1} \n  Last {n} Times: {t:.2f}s \n  Final Charges: {amount}')  # Add title, time, and charges

    # Second subplot for potential plot
    ax2 = fig.add_subplot(222)
    ax2.plot(range(len(potentials)), potentials, lw=2)
    ax2.set_xlim(0, len(potentials))
    ax2.set_ylim(min(potentials), max(potentials))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Potential Energy')
    ax2.set_title('Average Potential Energy vs Iteration (p/i^2)')
    ax2.text(0.5, 0.9, f'Decrease: {(last_pmin - pmin)}', transform=ax2.transAxes, ha='center')  # Add text for decrease in potential energy

    # Third subplot for 2D plot
    ax3 = fig.add_subplot(223, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = 0.01*le
    delta = 0.01*le  # Adjust this value as needed
    z_indices = np.where((xy[:, 2] >= z_values - delta) & (xy[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xy[z_indices, 0]
    y_values = xy[z_indices, 1]

    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        ax3.scatter(x_values, y_values, c='orange')  # Convert col to a numpy array
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title(f'2D Slices at Z = {z_values-delta, z_values+delta} \n  Points: {len(x_values)}')
        ax3.set_xlim(0, le)  # Set x-axis limits to 0 and le
        ax3.set_ylim(0, le)  # Set y-axis limits to 0 and le
        ax3.grid(True)
        
    # Fourth subplot for 2D plot
    ax4 = fig.add_subplot(224, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = 0.5*le
    delta = 0.01*le  # Adjust this value as needed
    z_indices = np.where((xy[:, 2] >= z_values - delta) & (xy[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xy[z_indices, 0]
    y_values = xy[z_indices, 1]

    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        ax4.scatter(x_values, y_values, c='orange')  # Convert col to a numpy array
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title(f'2D Slices at Z = {z_values-delta, z_values+delta} \n  Points: {len(x_values)}')
        ax4.set_xlim(0, le)  # Set x-axis limits to 0 and le
        ax4.set_ylim(0, le)  # Set y-axis limits to 0 and le
        ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f"C:/Users/surface/Onedrive/CloudyLake Programming/Product/charge_generator.png", dpi=300)
    plt.close(fig)

def main():
    global xyxy,n
    t1 = time.time()
    last_pmin=0
    average_potentials = []
    p0 = float('inf')
    for i in range(amount):
        xyxy, pmin = generate(xyxy)
        average_potentials.append(pmin)
        if (i+1)%n==0:
            t2 = time.time()
            t=t2-t1
            t1 = time.time()
            draw_3d(xyxy,i,t,pmin,last_pmin,p0,le,n,amount,average_potentials)
            last_pmin=pmin
    print(xyxy)
    print(pmin)

# 启动！
main()
