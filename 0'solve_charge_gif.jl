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
using Plots
function main()
    global xy
    global pmin
    t1 = now()
    anim = Animation()
    @animate for i in 1:n
        plt = plot(legend=false)
        xy, pmin = simulated_annealing(xy)
        if i % 300 == 0
            plt = plot(legend=false, size=(500, 500))
            scatter!(plt, xy[:,1], xy[:,2], color=:red, legend=false)
            hline!(plt, [0, le], color=:grey, linestyle=:dot, linewidth=1)
            vline!(plt, [0, le], color=:grey, linestyle=:dot, linewidth=1)
            xlims!(plt, -10, le+10)
            ylims!(plt, -10, le+10)
            title!(plt, "location of charge\nn=$i")
            frame(anim)
        end
    end
    t2 = now()
    println(xy)
    println(pmin)
    println(t2-t1)
    gif(anim, "charge_animation.gif", fps = 3)
end

#发车！
main()


