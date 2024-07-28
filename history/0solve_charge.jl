using Random
using Plots
using Dates
# 设置网格和电荷数
println("dpi=?*?")
le = parse(Int, readline())
println("charge amount?")
amount = parse(Int, readline())
println("times?")
n = parse(Int, readline())
xy = hcat(rand(0:le+1, amount), rand(0:le+1, amount))

# 计算势能
function solve_p(xy)
    pi = zeros(amount)
    for i1 in 1:amount
        for i2 in 1:amount
            d = sqrt((xy[i1,1]-xy[i2,1])^2 + (xy[i1,2]-xy[i2,2])^2)
            if d != 0
                pi[i1] += 1.0/d
            end
        end
    end
    return sum(pi)
end

# 生成新坐标时排除和其他坐标一样的函数
function randintxy_except(x, y, xy)
    for _ in 1:100
        a = rand(x:y)
        b = rand(x:y)
        if all(xy[:,1] .!= a .|| xy[:,2] .!= b)
            return [a, b]
        end
    end
    return [0, 0]
end

# 模拟退火算法
function simulated_annealing(xy)
    current_p = solve_p(xy)
    new_xy = copy(xy)
    i = rand(1:amount-1)
    new_xy[i,:] = randintxy_except(0, le, xy)
    new_p = solve_p(new_xy)
    if new_p < current_p
        xy = new_xy
        current_p = new_p
    end
    return xy, current_p
end

# 主函数
function main()
    global xy
    global pmin
    t1 = now()
    for i in 1:n
        xy, pmin = simulated_annealing(xy)
        if i % 100 == 0
            scatter(xy[:,1], xy[:,2], color=:red, legend=false)
            hline!([0, le], color=:grey, linestyle=:dot, linewidth=1)
            vline!([0, le], color=:grey, linestyle=:dot, linewidth=1)
            xlims!(-10, le+10)
            ylims!(-10, le+10)
            title!("location of charge\nn=$i")
            display(plot!(size=(600, 600)))
        end
    end
    t2 = now()
    #输出一下附加产品
    println(xy)
    println(pmin)
    println(t2-t1)
end
    
#发车！
main()


