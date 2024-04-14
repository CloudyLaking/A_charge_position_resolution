import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

##退火方法3D网格中求解数个电荷电势最低分布##
#元数据获取函数
def origin():
    # 设置网格大小与椭球大小
    le= 1
    a=0.5
    b=0.5
    c=0.0

    # 电荷数
    print("charges amount?")
    amount = int(input())
    # 多少次输出一次图像
    print("interval times?")
    n = int(input())
    #每次测量几次
    times=int(input('measurement times?'))
    #分成多少组数据
    datatimes=int(input('data times?'))
    #返回
    return le,amount,n,a,b,c,times,datatimes


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

#计算椭球顶点高斯曲率的函数
def curvature(a, b, c):
    return np.square(c) / np.multiply(np.square(a) ,np.square(b))

# 范围内正态生成一个新坐标的函数
def randintxyz_except(le, xyz, i, a0,b0,c0):
    lenxyz=len(xyz)
    for _ in range(100):
        a = np.clip(np.random.normal(xyz[i, 0], 0.2), -a0, a0)
        b = np.clip(np.random.normal(xyz[i, 1], 0.2), -b0, b0)
        c = np.clip(np.random.normal(xyz[i, 2], 0.2), -c0, c0)
        if isin([a,b,c],a0,b0,c0):
            return np.array([a, b, c])
    return np.array([xyz[i,0], xyz[i,1], xyz[i,2]])

#随机生成一个范围内新坐标的函数
def randintxyz_generate(le, xyz, a0,b0,c0):
    lenxyz=len(xyz)
    for _ in range(100):
        a = np.random.uniform(-a0, a0)
        b = np.random.uniform(-b0, b0)
        c = np.random.uniform(-c0, c0)
        if isin([a,b,c],a0,b0,c0):
            return np.array([a, b, c])
    return np.array([0,0,0]) #为了防止出错还是弄了一个  

# 模拟退火算法一次
def simulated_annealing(le,amount,xyz,ii,current_p,a,b,c):
    #一次更新n个
    n=int(0.05*amount)
    for _ in range(n):
        new_xyz = np.copy(xyz)
        i = random.randint(0, len(xyz)-1)
        new_xyz[i,:] = randintxyz_except(le,xyz,i,a,b,c)
    #算新势能
    new_p = solve_p(new_xyz)
    #蒙特卡洛判决
    if new_p < current_p:
        return new_xyz, new_p
    return xyz, current_p
    
#画图函数
def draw_3d(xyz,i,t,pmin,last_pmin,p0,le,n,amount,potentials,a0,b0,c0):
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
    z_values = 0.95*c0
    delta = 0.05*c0  # Adjust this value as needed
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

    #记录个数
    data0 = len(x_values) 
    return data0

def main(le,amount,n,a,b,c):
    #初始化坐标
    xyz = np.empty((0, 3))
    for _ in range(amount):
        xyz=np.append(xyz,[randintxyz_generate(le, xyz, a,b,c)] ,axis=0)

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
        xyz, pmin = simulated_annealing(le,amount,xyz,i,pmin,a,b,c)
        potentials.append(pmin)
        #隔段画图
        if (i+1)%n==0:
            #计时
            t2 = time.perf_counter()
            t=t2-t1
            t1 = time.perf_counter()
            #画图
            data0=draw_3d(xyz,i,t,pmin,last_pmin,p0,le,n,amount,potentials,a,b,c)
            #记下上次最小值
            last_pmin=pmin
            if potentials[int(i/2)]-pmin<(potentials[0]-pmin)/15:
                break
    print(pmin)
    return data0

def mainmain():
    #获取初始参数
    le,amount,n,a,b,c,times,datatimes=origin()
    #记录数据组
    data=np.empty((datatimes,times))
    datay=np.empty((datatimes))
    curva=np.empty((datatimes))
    for _ in range(datatimes):
        #得到c
        c=(_+1)*(1/datatimes)
        #计算对应高斯曲率
        curva[_]=curvature(a,b,c)
        #主循环试验
        for __ in range(times):
            start_time = time.perf_counter()
            data0=main(le,amount,n,a,b,c)
            data[_,__]=data0
            print(data0)
            end_time = time.perf_counter()
            t = end_time - start_time
            #print检验
            print("Finished!")
            print(f'c: {c}, measured time:{__+1}/{times}, present data: {int(data[_,__])}')
            print(f'Time cost: {t} seconds')
            print('\n')
        print('curvature={curva[-1]}\n')

        #赋值
        datay[_]=np.mean(data[_])

        # 作图模块
        for ___ in range(1):
            #每一组data的平均值用红色点显示
            plt.scatter(np.power(curva, 1/4), datay, color='red', marker='o', s=40)

            #把所有data数据点用灰色点显示
            curva_expand = np.repeat(curva, times, axis=0)
            plt.scatter(np.power(curva_expand, 1/4), data.ravel(), color='gray', marker='o', s=10)

            #标题
            plt.xlabel('K^(1/4)')
            plt.ylabel('sigma')
            plt.title('Relationship between sigma and K^(1/4)')

            # 线性回归
            X = np.power(curva, 1/4).reshape(-1, 1)
            y = datay.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            plt.plot(np.power(curva, 1/4), y_pred, color='blue', linewidth=2)

            # 延长趋势线
            x_extend = np.linspace(0, 2, 100).reshape(-1, 1)
            y_extend = reg.predict(x_extend)
            plt.plot(x_extend, y_extend, color='blue', linestyle='--', linewidth=2)
            
            # 计算R方
            r2 = r2_score(y, y_pred)
            plt.text(0.1, 0.9, f'R^2 = {r2:.2f}', transform=plt.gca().transAxes)

            # 显示基本信息
            info = f'Amount: {amount}  \na: {a}  \nb: {b}  \nC:0.1-1.0  \nmeasurement Times: {times}  \nDataTimes: {1/datatimes}'
            plt.text(0.6, 0.1, info, transform=plt.gca().transAxes)

            #图基本设置
            plt.grid(True)
            plt.xlim(0, 2)  # Set x-axis limits
            plt.savefig(f"C:/Users/lihui/OneDrive/CL/OneDrive/CloudyLake Programming/Product/location.png", dpi=300)
            plt.close()

    print(data)
    print(datay)

    
# 启动！
mainmain()