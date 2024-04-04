import numpy as np
import matplotlib as mpl
import time
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

# 3D网格中枚举求解数个电荷电势最低分布

import matplotlib.pyplot as plt

# 设置网格分辨率
print("dpi=?*?")
le = int(input())
# 电荷数
print("charges amount?")
amount = int(input())

# 生成初坐标
xy = np.column_stack((np.random.randint(0, le, amount), np.random.randint(0, le, amount), np.random.randint(0, le, amount)))
n=1000

# 计算势能
def solve_p(xy):
    # 构建 k-d 树
    tree = cKDTree(xy)
    # 查询树以找到每个点的最近邻居
    distances, indices = tree.query(xy, k=2)
    # 第一个返回的邻居是点自身，所以我们取第二个
    distances = distances[:, 1]
    epsilon = 1e-8  # 避免除以零
    # 计算势能并求和
    potentials = np.sum(1.0 / (distances + epsilon))
    return potentials

# 画图函数
def draw_3d(xy, t, pmin, le, amount):
    fig = plt.figure(figsize=(6, 6))
    cm = plt.get_cmap("coolwarm")  # 色图
    # 画数据
    col = [cm(float(xy[i, 2]) / (le)) for i in range(amount)]
    ax = fig.add_subplot(111, projection='3d')
    # Generate colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=le)
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

    # 画图
    ax.view_init(15, 60, 0)
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=col, depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"location of {amount} charges, t={n + 1} \n potential:{pmin} \n time:{t:.2f}s")
    
    plt.tight_layout()
    plt.savefig("C:/Users/surface/Onedrive/CloudyLake Programming/Product/charge.png", dpi=300)
    plt.close(fig)

def traverse_coordinates(xy, le):
    best_xy = np.copy(xy)
    best_pmin = solve_p(xy)
    for n in range(xy.shape[0]):
        for x in range(le):
            for y in range(le):
                for z in range(le):
                    xy_copy = np.copy(xy)  # 创建副本以避免修改原始数组
                    xy_copy[n, :] = [x, y, z]  # 重新设置坐标
                    pmin = solve_p(xy_copy)
                    if pmin < best_pmin:
                        best_xy = np.copy(xy_copy)
                        best_pmin = pmin
    return best_xy, best_pmin


# 主函数
def main():
    global xy, amount
    t1 = time.time()
    best_xy, best_pmin = traverse_coordinates(xy, le)
    t2 = time.time()
    t = t2 - t1
    t1 = time.time()
    draw_3d(best_xy, t, best_pmin, le, amount)  #  画图
    print(best_xy)

# 启动！
main()