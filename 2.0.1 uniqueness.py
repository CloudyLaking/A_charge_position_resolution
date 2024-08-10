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
import numba
import matplotlib.pyplot as plt
from itertools import combinations, product
import os

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
@numba.njit
def solve_p(xyxy):
    # 计算点对之间的距离
    distances = np.sqrt(np.sum((xyxy[:, np.newaxis] - xyxy)**2, axis=-1))
    # 将对角线上的元素设置为无穷大，以排除自距离
    np.fill_diagonal(distances, np.inf)    
    # 计算势能
    return np.sum(np.divide(1.0 , distances))

#判断坐标xyz是否在abc椭球范围的函数

def isin(xyz,a0,b0,c0):
    return np.sum(np.divide(xyz , np.array([a0, b0, c0]))**2) <= 1

#计算椭球（1/ab）的函数

def curvature(a, b, c):
    return np.divide(1,np.multiply(a,b))


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
def simulated_annealing(xyz,current_p,a,b,c, m,step,T):
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

def draw_3d(xyz,xyz0,i,t,pmin,last_pmin,le,n,amount,potentials,a0,b0,c0):
    fig = plt.figure(figsize=(15, 10))
    cm = plt.get_cmap("binary")  # Set colormap to binary (black and white)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    
    # First subplot for 3D plot
    col = np.sqrt((xyz0[:, 0]/a0)**2 + (xyz0[:, 1]/b0)**2 + (xyz0[:, 2]/c0)**2)      # Normalize the distance to [0, 1]
    col = (col - np.min(col)) / (np.max(col) - np.min(col))
    
    # Convert the distance to colors
    col = np.array([(1-col[i], 1-col[i], 1-col[i]) for i in range(amount)])
    
    # Draw a cube
    r = [-round(a0, 2), round(a0, 2)] if a0 > c0 else [-round(c0, 2), round(c0, 2)]
    
    ax1 = fig.add_subplot(231, projection='3d')  # Modify the subplot number to 221
    # Remove grid lines
    ax1.grid(False)
    # Draw a cube
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax1.plot3D(*zip(s, e), color="black", linewidth=0.5)

    # Set aspect ratio
    ax1.set_box_aspect([1,1,1])

    # Draw scatter plot
    ax1.view_init(15, 60, 0)
    ax1.scatter(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], c=col, depthshade=True,s=10)
    # Set the title for ax1
    ax1.set_title('(a) 电荷初始分布状态\n距离原点较远的电荷颜色更深', loc='left')
    # Set the number of ticks on the x, y, and z axes to 2
    ax1.set_xticks(r)
    ax1.set_yticks(r)
    ax1.set_zticks(r)

    # Set the tick labels to the minimum and maximum values
    ax1.set_xticklabels(r)
    ax1.set_yticklabels(r)
    ax1.set_zticklabels(r)


    # First subplot for 3D plot
    col = np.sqrt((xyz[:, 0]/a0)**2 + (xyz[:, 1]/b0)**2 + (xyz[:, 2]/c0)**2)      # Normalize the distance to [0, 1]    # Normalize the distance to [0, 1]
    col = (col - np.min(col)) / (np.max(col) - np.min(col))

    # Convert the distance to colors
    col = np.array([(1-col[i], 1-col[i], 1-col[i]) for i in range(amount)])
    
    ax2 = fig.add_subplot(232, projection='3d')  # Modify the subplot number to 221
    # Remove grid lines
    ax2.grid(False)
    # Set aspect ratio
    ax2.set_box_aspect([1,1,1])
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax2.plot3D(*zip(s, e), color="black", linewidth=0.5)
    # Draw scatter plot
    ax2.view_init(15, 60, 0)
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, depthshade=True,s=10)
    # Set the title for ax1
    ax2.set_title('(b) 电荷模拟分布状态\n距离原点较远的电荷颜色更深', loc='left')
    # Set the number of ticks on the x, y, and z axes to 2
    ax2.set_xticks(r)
    ax2.set_yticks(r)
    ax2.set_zticks(r)

    # Set the tick labels to the minimum and maximum values
    ax2.set_xticklabels(r)
    ax2.set_yticklabels(r)
    ax2.set_zticklabels(r)

    # 3rd subplot for potential plot
    ax3 = fig.add_subplot(233)
    if len(potentials)<500:
        ax3.plot(range(len(potentials)), potentials, lw=2, color='black')
        ax3.set_xlim(0, len(potentials))
        ax3.set_ylim(min(potentials), max(potentials))
    else:
        ax3.plot(range(1000, len(potentials)), potentials[1000:], lw=2, color='black')
        ax3.set_xlim(0, len(potentials)-1000)
        ax3.set_ylim(min(potentials), max(potentials[1000:]))
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('势能')
    ax3.set_title('(c) 势能与迭代次数关系', loc='left')
    ax3.grid(True)

    #4th subplot for 2D plot
    ax4 = fig.add_subplot(234, aspect='equal')  # Set aspect ratio to 'equal'
    x_values = np.where((xyz[:, 0] >= -0.05*a0) & (xyz[:, 0] <= 0.05*a0))[0]  # Select rows where x is in [-0.05*a0, 0.05*a0]
    y_values = xyz[x_values, 1]
    z_values = xyz[x_values, 2]

    if y_values.size>0:  # Check if y_values and z_values are not empty
        ax4.scatter(y_values, z_values, c='black',s=5)  # Plot the points
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('(d)x = ± 0.05 a 2D截面内的电荷分布', loc='left')
        ax4.set_xlim(-r[1], r[1])  # Set x-axis limits to -b0 and b0
        ax4.set_ylim(-r[1], r[1])  # Set y-axis limits to -c0 and c0
        ax4.grid(True)
        # Add ellipse
        ellipse = mpl.patches.Ellipse((0, 0), 2*b0, 2*c0, edgecolor='black', facecolor='none')
        ax4.add_patch(ellipse)
    
    #5th subplot for 2D plot
    ax5 = fig.add_subplot(235, aspect='equal')  # Set aspect ratio to 'equal'
    y_values = np.where((xyz[:, 1] >= -0.05*b0) & (xyz[:, 1] <= 0.05*b0))[0]  # Select rows where y is in [-0.05*b0, 0.05*b0]
    x_values = xyz[y_values, 0]
    z_values = xyz[y_values, 2]

    if x_values.size>0:  # Check if x_values and z_values are not empty
        ax5.scatter(x_values, z_values, c='black',s=5)  # Plot the points
        ax5.set_xlabel('X')
        ax5.set_ylabel('Z')
        ax5.set_title('(e)y = ± 0.05 b 2D截面内的电荷分布', loc='left')
        ax5.set_xlim(-r[1], r[1])  # Set x-axis limits to -a0 and a0
        ax5.set_ylim(-r[1], r[1])  # Set y-axis limits to -c0 and c0
        ax5.grid(True)
        # Add ellipse
        ellipse = mpl.patches.Ellipse((0, 0), 2*a0, 2*c0, edgecolor='black', facecolor='none')
        ax5.add_patch(ellipse)
    
    # 6th subplot for 2D plot
    ax6 = fig.add_subplot(236, aspect='equal')  # Set aspect ratio to 'equal'
    z_values = 0
    delta = 0.05  # Adjust this value as needed
    z_indices = np.where((xyz[:, 2] >= z_values - delta) & (xyz[:, 2] <= z_values + delta))[0]  # Select rows where z is in [z_values - delta, z_values + delta]
    x_values = xyz[z_indices, 0]
    y_values = xyz[z_indices, 1]

    if x_values.size>0:  # Check if x_values and y_values are not empty
        ax6.scatter(x_values, y_values, c='black',s=5)  # Plot the points
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_title('(f)z = ± 0.05 c 2D截面内的电荷分布', loc='left')
        ax6.set_xlim(-le, le)  # Set x-axis limits to 0 and le
        ax6.set_ylim(-le, le)  # Set y-axis limits to 0 and le
        ax6.grid(True)
        # Add ellipse
        ellipse = mpl.patches.Ellipse((0, 0), 2*a0, 2*b0, edgecolor='black', facecolor='none')
        ax6.add_patch(ellipse)
            

    decrease_rate = -np.diff(potentials)
    # 获取当前文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 生成图片的完整路径
    image_path = os.path.join(script_dir, f'charge_anneal.png')

    # 保存图片到指定路径
    plt.savefig(image_path, dpi=300)
    plt.close()

    # 计算在椭球表面1%一层的电荷个数
    def point_to_ellipse_distance(x, y, z, a, b, c):
        # 计算
        return np.sqrt((x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2))
    
    threshold1 = 0.98  # 阈值
    threshold2 = 0.985
    threshold3 = 0.99
    threshold4 = 0.995
    distances = np.array([point_to_ellipse_distance(x, y, z, a0, b0, c0) for x, y, z in xyz])
    data1 = np.sum(distances >= threshold1)/amount
    data2 = np.sum(distances >= threshold2)/amount
    data3 = np.sum(distances >= threshold3)/amount
    data4 = np.sum(distances >= threshold4)/amount

    decreasepoint=len([i for i in decrease_rate[-20000:] if i > 0])
    return data1,data2,data3,data4,decreasepoint


def main(le,amount,n,a,b,c,m,step,T0,v):
    #初始化坐标
    xyz = np.empty((0, 3))
    for _ in range(amount):
        xyz=np.append(xyz,[randintxyz_generate(xyz, a,b,c)] ,axis=0)
    xyz0=xyz
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
    for i in range(10000000):
        #循环动作
        xyz, pmin = simulated_annealing(xyz,pmin,a,b,c,m,step,T)
        potentials.append(pmin)
        #隔段画图
        if (i-1)%n==0:
            #计时
            t2 = time.perf_counter()
            t=t2-t1
            t1 = time.perf_counter()
            #画图
            data1,data2,data3,data4,decreasepoint=draw_3d(xyz,xyz0,i,t,pmin,last_pmin,le,n,amount,potentials,a,b,c)
            #记下上次最小值
            last_pmin=pmin
            #判断是否稳定
            diff1 = np.subtract(potentials[int(i/2)], pmin)
            diff2 = np.subtract(potentials[int(i/10)], pmin)
            threshold = np.divide(np.subtract(potentials[0], pmin), v)
            #开始收敛
            if diff1 < threshold*100 and diff2 < threshold*200 and diff1 > 0 and diff2 > 0:
                T = T * 0.4
                step=step*0.5 if step>step0 else step0
                print('Converging... T={}, step={}'.format(T, step))
            else:
                print('Annealing... T={}, step={}'.format(T, step))
            if  i>400000:
                break
    return data1,data2,data3,data4,pmin


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
    v=1000

    #记录数据组
    dataarray1=np.empty((datatimes,times))
    dataarray2=np.empty((datatimes,times))
    dataarray3=np.empty((datatimes,times))
    dataarray4=np.empty((datatimes,times))
    datay1=np.empty((datatimes))
    datay2=np.empty((datatimes))
    datay3=np.empty((datatimes))
    datay4=np.empty((datatimes))

    curva=np.empty((datatimes))
    for _ in range(datatimes):
        #顺序迭代c
        cc=(_)*(2/datatimes)+0.125
        c=1/cc
        #计算对应高斯曲率
        curva[_]=cc
        #正态生成坐标步长
        step=a*b*c*4

        #主循环试验
        for __ in range(times):
            start_time = time.perf_counter()
            data1,data2,data3,data4,pmin=main(le,amount,n,a,b,c,m,step,T0,v)
            dataarray1[_,__]=data1
            dataarray2[_,__]=data2
            dataarray3[_,__]=data3
            dataarray4[_,__]=data4
            end_time = time.perf_counter()
            t = end_time - start_time
            #print检验
            print("Finished!")
            print('c: {}, measured time:{}/{}'.format(c, __+1, times))
            print('Time cost: {} seconds'.format(t))
            print('pmin:{}'.format(pmin))
            print('\n')
            #取平均
            if __ == times-1:
                datay1[_]=np.mean(dataarray1[_])
                datay2[_]=np.mean(dataarray2[_])
                datay3[_]=np.mean(dataarray3[_])
                datay4[_]=np.mean(dataarray4[_])
        # 作图模块
            fig = plt.figure(figsize=(6, 6))
            #每一组data的平均值用红色点显示
            mask = [i for i in range(_) ]#筛查
            plt.scatter(curva[mask], datay1[mask], color='black', marker='^', s=40, label='2%')
            plt.scatter(curva[mask], datay2[mask], color='black', marker='s', s=40, label='1.5%')
            plt.scatter(curva[mask], datay3[mask], color='black', marker='o', s=40, label='1%')
            plt.scatter(curva[mask], datay4[mask], color='black', marker='D', s=40, label='0.5%')
            #连线
            plt.plot(curva[mask], datay1[mask], color='black', linestyle='-', linewidth=1)
            plt.plot(curva[mask], datay2[mask], color='black', linestyle='-', linewidth=1)
            plt.plot(curva[mask], datay3[mask], color='black', linestyle='-', linewidth=1)
            plt.plot(curva[mask], datay4[mask], color='black', linestyle='-', linewidth=1)
            plt.ylim(bottom=dataarray4[0].min()-0.05, top=1.05)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


            #标题
            plt.xlabel('1/c')
            plt.ylabel('η')
            plt.title('选取椭球不同厚度表面层时电荷占比η与椭球半轴长c的关系')
            # 获取当前文件的绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 生成图片的完整路径
            image_path = os.path.join(script_dir, f'charge_c.png')

            # 保存图片到指定路径
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(image_path, dpi=300)
            print(f'图片已保存至 {image_path}')
            plt.close()

        print('sigma:',datay1[_],datay2[_],datay3[_],datay4[_],'\n')

    print(datay1)
    print(datay2)
    print(datay3)
    print(datay4)

    
# 启动！
mainmain()
