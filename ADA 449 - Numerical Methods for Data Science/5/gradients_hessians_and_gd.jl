using ForwardDiff, Zygote, LinearAlgebra
using Plots

f(x) = x[1]^2 - x[1] * x[2]^2

# Compute the Hessian and gradient
H = Zygote.hessian(f, [-1.0, 1.0])[1]
G = Zygote.gradient(f, [-1.0, 1.0])[1]

println(H)
println(G)

# Assume that x
#x⋆ ∈ Ω is a an interior point. If:
#∇f (x⋆) = 0 && F(x⋆) > 0
#then x⋆ is a local minimizer of f .

#   Gradient Descent  #
# α is the learning rate.

function optimize(f::Function, x_init::AbstractVector; lr::AbstractFloat = 0.001, max_iter::Integer = 100, stopping_criterion::Float64 = 1e-2)
    for i in 1:max_iter
        g = Zygote.gradient(f, x_init)[1]
        x_init = x_init - lr * g
        if norm(g) < stopping_criterion
            @info "OK Boomer!!! you are done in $(i) steps"
            return x_init
        end
    end
    @warn "You are not done yet, try increasing the number of iterations"
    return x_init
end

optimize(f, [-1.0, 1.0], lr = 0.000001, max_iter = 1000000, stopping_criterion = 1e-2)
