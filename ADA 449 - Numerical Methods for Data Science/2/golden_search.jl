function golden(fun::Function, a::Real, b::Real; max_iter::Int = 100, ϵ::Float64 = 1e-3)
    r = (3-sqrt(5))/2

    x1 = a + r*(b-a)
    x2 = b - r*(b-a)
    

    f1 = fun(x1)
    f2 = fun(x2)

    k = 0

    for k in max_iter
        if (f1 > f2)
            a = x1
            x1 = x2
            f1 = f2
            x2 = r*a + (1-r)*b
            f2 = fun(x2) 
        else
            b = x2
            x2 = x1
            f2 = f1
            x1 = r*b + (1-r)*a
            f1 = fun(x1)
        end
        
        if abs(f1-f2) < ϵ
            println("Iteration Stopped in $(k) steps")
            return x1, f(α)
        end

    end
    @info "Algorithm did not converge correctly!!!"
    return x1, f(x1)

    

end



#############################

golden(x -> x^5-x, -5, 5; ϵ = 1e-100)