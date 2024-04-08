import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

##退火方法3D网格中求解数个电荷电势最低分布##

# 设置网格分辨率
le=1
#电荷数
print("charges amount?")
amount = int(input())
#多少次输出一次图像
print("interval times?")
n = int(input())

#定义导体的3D图形范围电荷随机生成函数


# 计算势能
def solve_p(xyxy):
    # 计算点对之间的距离
    distances = np.linalg.norm(xyxy[:, np.newaxis] - xyxy, axis=2)
    # 将对角线上的元素设置为无穷大，以排除自距离
    np.fill_diagonal(distances, np.inf)
    # 计算倒数之和
    return np.sum(1.0 / distances)

# 随机生成新坐标时排除和其他坐标一样的函数
def renew(xy, n=1):
    for _ in range(n):
        i = random.randint(0, amount-1)
        for _ in range(100):
            a = np.random.uniform(0, le)
            b = np.random.uniform(0, le)
            c = np.random.uniform(0, le)
            if all(((xy[ii,0] != a or xy[ii,1] != b or xy[ii,2]!=c) and 0<=a<=le and 0<=b<=le and 0<=c<=le) for ii in range(len(xy))):
                xy[i,:] = np.array([a, b, c])
    return xy

# 模拟退火算法一次  
def simulated_annealing(xyz,ii):
    current_p = solve_p(xyz)
    new_xyz = np.copy(xyz)
    new_xyz = renew(xyz, n=1)
    new_p = solve_p(new_xyz)
    if new_p < current_p:
        xyz = new_xyz
        current_p = new_p
        print(len(xyz),new_p)
    return xyz, current_p

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
    ax1.set_title(f'Iteration without anneal: {i+1} \n  Last {n} times: {t:.2f}s \n  Charges: {amount}')  # Add title, time, and charges

    # Second subplot for potential plot
    ax2 = fig.add_subplot(222)
    ax2.plot(range(len(potentials)), potentials, lw=2)
    ax2.set_xlim(0, len(potentials))
    ax2.set_ylim(min(potentials), max(potentials))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Potential Energy')
    ax2.set_title('Potential Energy vs Iteration')
    ax2.text(0.5, 0.9, f'Decrease: {last_pmin - pmin}\n present potential:{pmin}', transform=ax2.transAxes, ha='center')  # Add text for decrease in potential energy
    ax2.grid(True)

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
    plt.savefig(f"C:/Users/surface/Onedrive/CloudyLake Programming/Product/charge_non_anneal.png", dpi=300)
    plt.close(fig)

def main():
    #大参数
    global n,amount,le
    # 生成初坐标
    xy = np.column_stack((np.random.uniform(0, le, amount), np.random.uniform(0, le, amount), np.random.uniform(0, le, amount)))
    #计时
    t1 = time.perf_counter()
    #初始化
    last_pmin=0
    potentials = []
    p0=solve_p(xy)
    #循环
    for i in range(100000):
        #循环动作
        xy, pmin = simulated_annealing(xy,i)
        potentials.append(pmin)
        #隔段画图
        if (i+1)%n==0:
            #计时
            t2 = time.perf_counter()
            t=t2-t1
            t1 = time.perf_counter()
            #画图
            draw_3d(xy,i,t,pmin,last_pmin,p0,le,n,amount,potentials)
            #记下上次最小值
            last_pmin=pmin
    print(xy)
    print(pmin)

# 启动！
main()