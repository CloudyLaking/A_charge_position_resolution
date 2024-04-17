import numpy as np
import matplotlib as mpl
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from multiprocessing import Pool
from numba import jit
import matplotlib.pyplot as plt

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
    #多少次输出一次图像
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
    distances = np.sqrt(np.sum((xyxy[:, np.newaxis] - xyxy)**2, axis=-1))
    # 你的其他代码...
    # 将对角线上的元素设置为无穷大，以排除自距离
    np.fill_diagonal(distances, np.inf)
    # 计算倒数之和
    return np.sum(np.divide(1.0 , distances))

#判断坐标xyz是否在abc椭球范围的函数

def isin(xyz,a0,b0,c0):
    return np.sum(np.divide(xyz , np.array([a0, b0, c0]))**2) <= 1

#计算椭球（1/ab）的函数

def curvature(a, b, c):
    return np.divide(1,np.multiply(a,b))

#计算R方
def fit_model(X, y):
    return LinearRegression().fit(X, y)

# 范围内正态生成一个新坐标的函数

def randintxyz_except(xyz, i, a0 ,b0 ,c0, step):
    lenxyz=len(xyz)
    for _ in range(100):
        a = np.clip(np.random.normal(xyz[i, 0], step), -a0, a0)
        b = np.clip(np.random.normal(xyz[i, 1], step), -b0, b0)
        c = np.clip(np.random.normal(xyz[i, 2], step), -c0, c0)
        if isin([a,b,c],a0,b0,c0):
            return np.array([a, b, c])
    return np.array([xyz[i,0], xyz[i,1], xyz[i,2]])

#随机生成一个范围内新坐标的函数

def randintxyz_generate(xyz, a0,b0,c0):
    lenxyz=len(xyz)
    for _ in range(100):
        a = np.random.uniform(-a0, a0)
        b = np.random.uniform(-b0, b0)
        c = np.random.uniform(-c0, c0)
        if isin([a,b,c],a0,b0,c0):
            return np.array([a, b, c])
    return np.array([0,0,0]) #为了防止出错还是弄了一个  


# 模拟退火算法一次
def simulated_annealing(xyz,ii,current_p,a,b,c, m,step,T):
    new_xyz = np.copy(xyz)
    for _ in range(m):
        i = random.randint(0, len(xyz)-1)
        new_xyz[i,:] = randintxyz_except(xyz,i,a,b,c,step)
    #算新势能
    new_p = solve_p(new_xyz)
    #蒙特卡洛判决
    if new_p < current_p:
        return new_xyz, new_p
    elif np.random.rand() < np.exp(np.divide(np.subtract(current_p , new_p) , np.multiply(current_p,T))):
        return new_xyz, new_p
    return xyz, current_p
    
#画图函数

def draw_3d(xyz,i,t,pmin,last_pmin,le,n,amount,potentials,a0,b0,c0):
    fig = plt.figure(figsize=(10, 10))
    cm = plt.get_cmap("coolwarm")  # 色图

    # First subplot for 3D plot
    col = [cm((float(xyz[i,2]))/(le)) for i in range(amount)]
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
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, depthshade=True,s=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Iteration with anneal: {} \n  Last {} times: {:.2f}s \n  Charges: {}'.format(i+1, n, t, amount))  # Add title, time, and charges
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
    ax2.text(0.5, 0.9, 'Decrease: {}\n present potential:{}'.format(last_pmin - pmin, pmin), transform=ax2.transAxes, ha='center')  # Add text for decrease in potential energy
    ax2.grid(True)

    # Third subplot for 2D plot
    ax3 = fig.add_subplot(223, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = 0
    delta = 0.05  # Adjust this value as needed
    z_indices = np.where((xyz[:, 2] >= z_values - delta) & (xyz[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xyz[z_indices, 0]
    y_values = xyz[z_indices, 1]


    if x_values.size > 0 and y_values.size > 0:  # Check if x_values and y_values are not empty
        ax3.scatter(x_values, y_values, c='orange')  # Convert col to a numpy array
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        ax3.set_xlim(-le, le)  # Set x-axis limits to 0 and le
        ax3.set_ylim(-le, le)  # Set y-axis limits to 0 and le
        ax3.grid(True)
        # Add ellipse
        ellipse = mpl.patches.Ellipse((0, 0), 2*a0, 2*b0, edgecolor='red', facecolor='none')
        ax3.add_patch(ellipse)

        ax3.set_title('2D Slices at Z = {:.4f}, {:.4f} \n  Points: {}  '.format(z_values-delta, z_values+delta, len(x_values)))
            
        #True Fourth subplot for 2D plot
        # Second subplot for potential decrease rate plot
        ax5 = fig.add_subplot(224)
        decrease_rate = -np.diff(potentials)
        colors = ['red' if rate < 0 else 'green' if rate > 0 else 'blue' for rate in decrease_rate[:-1]] + ['green']
        ax5.scatter(range(1, len(decrease_rate)+1), decrease_rate, c=colors)
        if len(decrease_rate) > 20000:
            ax5.set_xlim(len(decrease_rate)-20000, len(decrease_rate))
            ax5.set_ylim(min(decrease_rate[-20000:]), max(decrease_rate[-20000:]))
        else:
            ax5.set_xlim(0, len(decrease_rate))
            ax5.set_ylim(min(decrease_rate), max(decrease_rate))
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Potential Decrease Rate')
        ax5.set_title('Potential Decrease Rate vs Iteration')
        ax5.grid(True)


    plt.tight_layout()
    plt.savefig("charge_anneal.png", dpi=300)
    plt.close(fig)

    #记录个数
    data0 = len(x_values) 
    return data0


def main(le,amount,n,a,b,c,m,step,T0,v):
    #初始化坐标
    xyz = np.empty((0, 3))
    for _ in range(amount):
        xyz=np.append(xyz,[randintxyz_generate(xyz, a,b,c)] ,axis=0)

    #初始化设置
    potentials = []
    p0=solve_p(xyz)
    last_pmin=p0
    pmin=p0
    step0=0.1*step
    #计时
    t1 = time.perf_counter()
    T=T0

    #循环
    for i in range(500000):
        #循环动作
        xyz, pmin = simulated_annealing(xyz,i,pmin,a,b,c,m,step,T)
        potentials.append(pmin)
        #隔段画图
        if (i+1)%n==0:
            #计时
            t2 = time.perf_counter()
            t=t2-t1
            t1 = time.perf_counter()
            #画图
            data0=draw_3d(xyz,i,t,pmin,last_pmin,le,n,amount,potentials,a,b,c)
            #记下上次最小值
            last_pmin=pmin
            #判断是否稳定
            diff1 = np.subtract(potentials[int(i/2)], pmin)
            diff2 = np.subtract(potentials[int(i/20)], pmin)
            threshold = np.divide(np.subtract(potentials[0], pmin), v)
            #开始收敛
            if diff1 < threshold*5 and diff2 < threshold*8 and diff1 > 0 and diff2 > 0:
                T = T * 0.75
                step=step*0.9 if step>step0 else step0
                print('Converging... T={}, step={}'.format(T, step))
            else:
                print('Annealing... T={}, step={}'.format(T, step))
            if (diff1 < threshold and diff2 < threshold and diff1 > 0 and diff2 > 0):
                break
    return data0,pmin


def mainmain():
    #获取初始参数
    le,amount,n,a,b,c,times,datatimes=origin()
    #核心控制参数：
    #每次移动数量
    pm=0.01
    m=int(pm*amount) if int(pm*amount)>0 else 1
    #初始温度
    T0=0.0002
    #稳态判据
    v=80

    #记录数据组
    data=np.empty((datatimes,times))
    datay=np.empty((datatimes))
    curva=np.empty((datatimes))
    for _ in range(datatimes):
        #顺序迭代c
        c=(_+1)*(1/datatimes)
        #计算对应高斯曲率
        curva[_]=1/c
        #正态生成坐标步长
        step=a*b*c*4
        #主循环试验
        for __ in range(times):
            start_time = time.perf_counter()
            data0,pmin=main(le,amount,n,a,b,c,m,step,T0,v)
            data[_,__]=data0
            end_time = time.perf_counter()
            t = end_time - start_time
            #print检验
            print("Finished!")
            print('c: {}, measured time:{}/{}, present data: {}'.format(c, __+1, times, int(data[_,__])))
            print('Time cost: {} seconds'.format(t))
            print('pmin:{}'.format(pmin))
            print('\n')

        #赋值
        datay[_]=np.mean(data[_])
        print('sigma:',datay[_],'\n')

        # 作图模块
        for ___ in range(1):
            #每一组data的平均值用红色点显示
            plt.scatter(curva, datay, color='red', marker='o', s=40)

            #把所有data数据点用灰色点显示
            curva_expand = np.repeat(curva, times, axis=0)
            plt.scatter(curva_expand, data.ravel(), color='gray', marker='o', s=10)

            #标题
            plt.xlabel('1/c')
            plt.ylabel('Sigma')
            plt.title('Relationship between sigma of A/B and 1/c')

            # 线性回归
            X = curva.reshape(-1, 1)
            y = datay.reshape(-1, 1)
            reg = fit_model(X, y)
            y_pred = reg.predict(X)
            plt.plot(curva, y_pred, color='blue', linewidth=2)

            # 延长趋势线
            x_extend = np.linspace(0, 2, 100).reshape(-1, 1)
            y_extend = reg.predict(x_extend)
            plt.plot(x_extend, y_extend, color='blue', linestyle='--', linewidth=2)
            
            # 计算R方
            r2 = r2_score(y, y_pred)
            plt.text(0.1, 0.9, 'R^2 = {:.2f}'.format(r2), transform=plt.gca().transAxes)

            # 显示基本信息
            info = 'Amount: {}  \na: {}  \nb: {}  \nC:{}-{}  \nMeasurement Times: {}  \nData division: {}'.format(amount, a, b, le/datatimes, le, times, 1/datatimes)
            plt.text(0.6, 0.1, info, transform=plt.gca().transAxes)

            #图基本设置
            plt.grid(True)
            plt.savefig("charge_c.png", dpi=300)
            plt.close()

    print(data)
    print(datay)

    
# 启动！
mainmain()