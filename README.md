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
初步优化结构
我们发现，电荷数对性能影响很大，而网格大小基本没有影响
我们使用np中的数据结构优化，减少了不需要的运行空间
最逆天的提升肯定发生在solve_p部分，通过广播的优化直接提升50倍速度
再换成barnes_hut算法后提升了1/3左右
然后用新电脑跑
*1.3.2*
把坐标改为浮点数再来
没什么吊用，但是还是把代码都改成浮点了
*1.3.3*
细致调整
删掉了输入dpi
加入了子图显示势能变化
最后看出来其实两个算法性能差不多
*1.3.4*
*1.3.5*
加入了z=0.1\0.5的剖面图
还尝试写了一个迭代求解
考虑到几乎不可能算出来遂罢

**1.4**
遗传算法重写对比
*1.4.1*
初步写完了遗传算法
*1.4.2*
md被骗了
那个还是个退火
我亲自来重写
但是结果比较抽象，波动得更厉害了


*1.5*
generator and 遗传算法改进 出结果了！
*1.5.1*
我们通过每一次加一个点的方法来写
让电荷自己找到最fit自己的位置
但是搞了半天还是错的
*1.5.2*
跑出来了
结果难以置信地完美
说明我们的退火真的还要优化
*1.5.3*
这个算法速度还有点慢，因为它的运行时间随着电荷增加
我们优化了算法结构，速度还是比较可观
可以运行到1000左右
总比完全遍历好多了
*1.5.4*
终于找到问题了
之前的solve_p函数只加了一个点的
意思就是说全错了
因为我们最开始的代码就是错的
那意思就是说这样整个运算时间要大大加长
但是结果出来了基本没问题！！
不过我们认为退火中温度其实是真的没有什么作用
反而是副作用，还要影响最终效果
*1.5.5*
为画图增加更多细节：
改成了平均势能
包括当前势能，关键点显示
*1.5.6*
大幅修改遗传算法
并且限制种群数量4
跑出的结果也很难蚌虽然好一些了
变成了超大幅度波动
*1.5.7*
一个新的退火温度设置
但是结果没有变化
不过好像快一些
不知道是什么因素
先放着
*1.6*规整化与性能提高，和1.5在一块属于
*1.6.1*
代码规整化
*1.6.2*
调好了随机生成的步长，取得显著效果
这个原理还不得而知
但是1000个的500次运行时间缩短到了20-30s附近
同时达到1.55的次数只用了1500左右
缩小到了五分之一
10000次就基本稳定了

*1.7*图形化与结果数据化，与论文报告接轨
此时已经写好论文框架了
*1.7.1*
空了电脑坏掉的一周再来
是时候要收尾阶段了
这一版本完成了椭球模型的搭建
*1.7.2*
将a固定为0.5，b固定为0.5，c在0.1-1.0之间变动测出10组三平均数据画图
改成了基本稳定了就取值
一共要跑30次
*1.7.3*
把次数提高再测试
显示误差线
做了很多人性化的优化
最后开跑数据
基本符合预期
*1.7.4*
为了加快速度得到更好结果删除了很多冗余步骤
虽然作用不大
*1.7.5*
但是突然想起一个问题
好像这样应该测量的是sigma与K^（1/4）~c^(-1/2)的关系
汗流浃背遂重测

*1.7.6*
新程序研究固定c时
a,b两点的曲率和K是否满足
