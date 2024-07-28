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
#元数据获取函数
def origin():
    # 设置网格大小与椭球大小
    le = 1
    a=float(input("a?"))
    b=float(input("b?"))
    c=float(input("c?"))
    a=0.4
    b=0.7
    c=0.9
    abc=[a,b,c]
    # 电荷数
    print("charges amount?")
    amount = int(input())
    # 多少次输出一次图像
    print("interval times?")
    n = int(input())
    #返回
    return le,amount,n,abc


# 计算势能
def solve_p(xyxy):
    # 计算点对之间的距离
    distances = np.linalg.norm(xyxy[:, np.newaxis] - xyxy, axis=2)
    # 将对角线上的元素设置为无穷大，以排除自距离
    np.fill_diagonal(distances, np.inf)
    # 计算倒数之和
    return np.sum(1.0 / distances)

#判断坐标xyz是否在abc椭球范围的函数
def isin(xyz,a0,b0,c0):
    return (xyz[0]/a0)**2+(xyz[1]/b0)**2+(xyz[2]/c0)**2<=1

# 范围内正态生成一个新坐标的函数
def randintxyz_except(le, xyz, i, abc):
    lenxyz=len(xyz)
    a0=abc[0]
    b0=abc[1]
    c0=abc[2]
    for _ in range(100):
        a = np.clip(np.random.normal(xyz[i, 0], 0.3), -le, le)
        b = np.clip(np.random.normal(xyz[i, 1], 0.3), -le, le)
        c = np.clip(np.random.normal(xyz[i, 2], 0.3), -le, le)
        if all(((xyz[ii,0] != a or xyz[ii,1] != b or xyz[ii,2]!=c))\
        for ii in range(lenxyz))\
        and isin([a,b,c],a0,b0,c0):
            return np.array([a, b, c])
    return np.array([xyz[i,0], xyz[i,1], xyz[i,2]])

#随机生成一个范围内新坐标的函数
def randintxyz_generate(le, xyz, abc):
    lenxyz=len(xyz)
    a0=abc[0]
    b0=abc[1]
    c0=abc[2]
    for _ in range(100):
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        c = np.random.uniform(-1, 1)
        if all(((xyz[ii,0] != a or xyz[ii,1] != b or xyz[ii,2]!=c))\
        for ii in range(lenxyz))\
        and isin([a,b,c],a0,b0,c0):
            return np.array([a, b, c])
    return np.array([xyz[0,0], xyz[1,1], xyz[1,2]]) #为了防止出错还是弄了一个  

# 模拟退火算法一次
def simulated_annealing(le,xyz,ii,current_p,abc):
    #一次更新n个
    n=10
    for _ in range(n):
        new_xyz = np.copy(xyz)
        i = random.randint(0, len(xyz)-1)
        new_xyz[i,:] = randintxyz_except(le,xyz,i,abc)
    #算新势能
    new_p = solve_p(new_xyz)
    #蒙特卡洛判决
    if new_p < current_p:
        return new_xyz, new_p
    return xyz, current_p
    
#画图函数
def draw_3d(xyz,i,t,pmin,last_pmin,p0,le,n,amount,potentials,abc):
    a0=abc[0]
    b0=abc[1]
    c0=abc[2]
    fig = plt.figure(figsize=(10, 10))
    cm = plt.get_cmap("coolwarm")  # 色图

    # First subplot for 3D plot
    col = [cm(float(xyz[i,2])/(le)) for i in range(amount)]
    ax1 = fig.add_subplot(221, projection='3d')  # Modify the subplot number to 221
    # Generate colorbar
    norm = mpl.colors.Normalize(vmin=-le, vmax=le)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5)
    # Remove grid lines
    ax1.grid(False)
    #draw
    ax1.view_init(15, 60, 0)
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, depthshade=True,s=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Iteration with anneal: {i+1} \n  Last {n} times: {t:.2f}s \n  Charges: {amount}')  # Add title, time, and charges
    # Add center axis
    ax1.plot([0], [0], [0], marker='o', markersize=5, color='red')
    ax1.set_xlim(-le, le)
    ax1.set_ylim(-le, le)
    ax1.set_zlim(-le, le)

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
    z_values = 0*le
    delta = 0.04*le  # Adjust this value as needed
    z_indices = np.where((xyz[:, 2] >= z_values - delta) & (xyz[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xyz[z_indices, 0]
    y_values = xyz[z_indices, 1]

    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        ax3.scatter(x_values, y_values, c='orange')  # Convert col to a numpy array
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title(f'2D Slices at Z = {z_values-delta, z_values+delta} \n  Points: {len(x_values)}')
        ax3.set_xlim(-le, le)  # Set x-axis limits to 0 and le
        ax3.set_ylim(-le, le)  # Set y-axis limits to 0 and le
        ax3.grid(True)
        # Add ellipse
        ellipse = mpl.patches.Ellipse((0, 0), 2*a0, 2*b0, edgecolor='red', facecolor='none')
        ax3.add_patch(ellipse)
        
    # Fourth subplot for 2D plot
    ax4 = fig.add_subplot(224, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = (c0-0.04)*le
    delta = 0.04*le  # Adjust this value as needed
    z_indices = np.where((xyz[:, 2] >= z_values - delta) & (xyz[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xyz[z_indices, 0]
    y_values = xyz[z_indices, 1]

    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        ax4.scatter(x_values, y_values, c='orange')  # Convert col to a numpy array
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title(f'2D Slices at Z = {z_values-delta, z_values+delta} \n  Points: {len(x_values)}')
        ax4.set_xlim(-le, le)  # Set x-axis limits to 0 and le
        ax4.set_ylim(-le, le)  # Set y-axis limits to 0 and le
        ax4.grid(True)
        # Add ellipse high
        aup=a0*np.sqrt(1-((z_values+delta)/c0)**2)
        bup=b0*np.sqrt(1-((z_values+delta)/c0)**2)
        ellipseup = mpl.patches.Ellipse((0, 0), 2*aup, 2*bup, edgecolor='red', facecolor='none')
        # Add ellipse low
        adown=a0*np.sqrt(1-((z_values-delta)/c0)**2)
        bdown=b0*np.sqrt(1-((z_values-delta)/c0)**2)
        ellipsdown = mpl.patches.Ellipse((0, 0), 2*adown, 2*bdown, edgecolor='blue', facecolor='none')

        ax4.add_patch(ellipseup)
        ax4.add_patch(ellipsdown)

    plt.tight_layout()
    plt.savefig(f"C:/Users/lihui/OneDrive/CL/OneDrive/CloudyLake Programming/Product/charge_anneal.png", dpi=300)
    plt.close(fig)

def main():
    #获取初始参数
    le,amount,n,abc=origin()
    #初始化坐标
    xyz = np.empty((0, 3))
    for _ in range(amount):
        xyz=np.append(xyz,[randintxyz_generate(le, xyz, abc)] ,axis=0)
    #初始化设置
    potentials = []
    p0=solve_p(xyz)
    last_pmin=p0
    pmin=p0
    #计时
    t1 = time.perf_counter()
    #循环
    for i in range(100000):
        #循环动作
        xyz, pmin = simulated_annealing(le,xyz,i,pmin,abc)
        potentials.append(pmin)
        #隔段画图
        if (i+1)%n==0:
            #计时
            t2 = time.perf_counter()
            t=t2-t1
            t1 = time.perf_counter()
            #画图
            draw_3d(xyz,i,t,pmin,last_pmin,p0,le,n,amount,potentials,abc)
            #记下上次最小值
            last_pmin=pmin
    print(xyz)
    print(pmin)

# 启动！
main()
