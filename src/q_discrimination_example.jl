#=
q_discrimination_example:
- Julia version: 1.10.5
- Author: mariakvashchuk
- Date: 2024-10-06
=#

using JuMP
import LinearAlgebra
import SCS

N, d = 2, 2

function random_state(d)
    x = randn(ComplexF64, (d, d))
    y = x * x'
    return LinearAlgebra.Hermitian(y / LinearAlgebra.tr(y))
end

# ρ = [random_state(d) for i in 1:N]
 ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]]
# ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
#  ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
# ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
# ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

model = Model(SCS.Optimizer)
set_silent(model)

E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

@constraint(model, sum(E) == LinearAlgebra.I)

@objective(
    model,
    Max,
    sum(real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N) / N,
)

optimize!(model)
@assert is_solved_and_feasible(model)
solution_summary(model)

objective_value(model)

0.5 + 0.25 * sum(LinearAlgebra.svdvals(ρ[1] - ρ[2]))

solution = [value.(e) for e in E]