using JuMP
import LinearAlgebra
import SCS


function qubit_bloch_sphere(theta, phi)
    return [cos(theta / 2), sin(theta / 2) * exp(im * phi)]
end

function iterate_over_disc(start = 0, stop = 1, step = 0.1, phi = 0)
    start += step
    result = []
    for idx in start:step:stop
        result = [result; [qubit_bloch_sphere(idx * 2 * pi, phi)]]
        # println("Result:\n", result)
    end
    return result
end

function solve_sdp(rho)
    N, d = length(rho), size(rho[1], 1)
    model = Model(SCS.Optimizer)
    set_silent(model)

    E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(E) == LinearAlgebra.I)

    @objective(
        model,
        Max,
        sum(real(LinearAlgebra.tr(rho[i] * E[i])) for i in 1:N) / N,
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)
    solution_summary(model)

    ov = objective_value(model)

    0.5 + 0.25 * sum(LinearAlgebra.svdvals(rho[1] - rho[2]))

    solution = [value.(e) for e in E]

    return [ov, solution]
end

function greedy_three_qubits(start = 0, stop = 1, step = 0.01)
    best_prob = 0
    best_rho = []
    best_povm = []
    values = iterate_over_disc(start, stop, step)
    density_matrices = [v * v' for v in values]
    rho0 = [1 0 ; 0 0]
    for i in 1:length(values) - 1
        rho1 = density_matrices[i]
        for j in i+1:length(values)
            rho2 = density_matrices[j]
            result = solve_sdp([rho0, rho1, rho2])
            # println("Result for ", values[i], " and ", values[j], ":")
            # println("Probability of success: ", result[1])
            # println("Solution:\n", result[2], "\n")
            if result[1] > best_prob
                best_prob = result[1]
                best_rho = [rho0, rho1, rho2]
                best_povm = result[2]
            end
        end
    end
    println("Best probability: ", best_prob)
    println("Best density matrices:\n", best_rho, "\n")
    println("Best POVM:\n", best_povm, "\n")
    return [best_prob, best_povm]
end

# result = solve_sdp([[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]])
result = greedy_three_qubits()
# println(result)
