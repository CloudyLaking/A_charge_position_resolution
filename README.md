# A_charge_position_resolution

**1.0**
*1.0.1 2024/3/23*
初始化
**1.1**
*1.1.1 2024/3/24*
把随机数生成范围限制在邻域，但是并没有什么作用
*1.1.2 2024/3/25*
考虑到性能问题决定还是先用单图运行
完善了单图的代码，并且改进了限制生成点的方法，提高了性能
还有做了规划
*1.1.3 2024/3/25*
try：每一次退火随机算完sqrt(n)个点，取最小,共sqrt(n)次
得到100 100 1000下可以只用16s n=10000只用150s
但是！
结果很搞笑
如算
每一个粒子都很自私，没有什么变化

**1.2**3D
*1.2.1 2024/3/26*
3D化初版本了
*1.2.2 2024/3/26*
优化了很多出图的细节
尤其重要的是把出图方式改为了设置出图周期而不是出图数量
图片通过保存的形式更新
*1.2.3 2024/3/26*
终于加入了温度概念
进行了更加完善的退火算法
*1.2.4 2024/3/27*
退火算法改进，因为跑了一晚上跑出来是个平面
*1.2.5 2024/3/28*
算法小改进
*1.2.6 2024/3/28*
增加了一个无退火的作为对照
因为跑出来结果非常尴尬
退火的算法可能写错了
到最后还在活动
不过无退火也没法退到四个角
非常奇怪

**1.3**架构重写性能改善
*1.3.1*
