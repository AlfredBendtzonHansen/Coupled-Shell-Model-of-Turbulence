using DifferentialEquations,Plots,LinearAlgebra,Random,DynamicalSystems, LaTeXStrings, Statistics,Distributions,JLD2,ProgressLogging

#JLD2.@load "Data/3d2d_t1-2_dt5e7_k01_L2_nu0.jld2" sol
#u0 = sol.u[end,:][1]
N = 20#Number of shells

λ = 2# lambda in k_n = k_0*lambda^n, canonically 2 for GOY and Sabra model
K0 = 1 # k_0 as above
kn = K0*λ.^(1:N) # kn's precomputed

u0¹ = zeros(ComplexF64,N).*kn.^(-1/3) 
u0² = zeros(ComplexF64,N).*kn.^(-1/3) 

#u0¹ = ones(ComplexF64,N).*kn.^(-1/3) 
#u0² = ones(ComplexF64,N).*kn.^(-1/3) 

ν = 5e-7# Kinetic viscosity

fn = zeros(ComplexF64,N) #Forcing
fn[7:8] .= (1 + 1*im)*0.1# Only small wave nr (large scale) should force to obser cascading and intermittency

kn = [0,0,kn...,0,0]
u0 = [0,0,u0¹...,0,0,u0²...,0,0]
du = zeros(ComplexF64, size(u0))

idx1_1 = 3
idx1_N = N+2
idx2_1 = idx1_N+3
idx2_N = idx1_N + N + 2

ϵ1 = 0.9
ϵ2 = 1.1
ϵv = 0.9
ϵvar1 = 1.1
ϵvar2 = 1.1
ϵul = 0.9
ϵur = 0.9
ϵll = 0.9
ϵlr = 0.9

p = tuple(kn, ϵ1, ϵ2, ϵv, ϵvar1, ϵvar2, ϵul, ϵur, ϵll, ϵlr, fn, ν, idx1_1, idx1_N,idx2_1, idx2_N)

function Sabra_coupled!(du, u, p, t)
    kn, ϵ1, ϵ2, ϵv, ϵvar1, ϵvar2, ϵul, ϵur, ϵll, ϵlr, fn, ν, idx1_1, idx1_N, idx2_1, idx2_N = p

    Γ_sabra_1 = conj(u[idx1_1+1:idx1_N+1]).*u[idx1_1+2:idx1_N+2] .- (ϵ1/λ).*conj(u[idx1_1-1:idx1_N-1]).*u[idx1_1+1:idx1_N+1] - ((ϵ1-1)/λ^2).*u[idx1_1-1:idx1_N-1].*u[idx1_1-2:idx1_N-2]
    Γ_v_1     = conj(u[idx2_1+1:idx2_N+1]).*u[idx1_1+2:idx1_N+2] .- (ϵv/λ).*conj(u[idx2_1-1:idx2_N-1]).*u[idx2_1+1:idx2_N+1] - ((ϵv-1)/λ^2).*u[idx2_1-1:idx2_N-1].*u[idx1_1-2:idx1_N-2]
    Γ_var1_1  = conj(u[idx1_1+1:idx1_N+1]).*u[idx2_1+2:idx2_N+2] .- (ϵvar1/λ).*conj(u[idx1_1-1:idx1_N-1]).*u[idx2_1+1:idx2_N+1] - ((ϵvar1-1)/λ^2).*u[idx2_1-1:idx2_N-1].*u[idx2_1-2:idx2_N-2]
    Γ_var2_1  = conj(u[idx2_1+1:idx2_N+1]).*u[idx2_1+2:idx2_N+2] .- (ϵvar2/λ).*conj(u[idx2_1-1:idx2_N-1]).*u[idx1_1+1:idx1_N+1] - ((ϵvar2-1)/λ^2).*u[idx1_1-1:idx1_N-1].*u[idx2_1-2:idx2_N-2]
    Γ_box_1   = ϵul.*u[idx2_1:idx2_N].*conj(u[idx2_1+1:idx2_N+1]) .+ (ϵur/λ).*u[idx2_1:idx2_N].*conj(u[idx2_1-1:idx2_N-1]) .+ ϵll.*u[idx2_1:idx2_N].*u[idx1_1+1:idx1_N+1] .+ (ϵlr/λ).*u[idx2_1:idx2_N].*u[idx1_1-1:idx1_N-1] 

    Γ_sabra_2 = conj(u[idx2_1+1:idx2_N+1]).*u[idx2_1+2:idx2_N+2] .- (ϵ2/λ).*conj(u[idx2_1-1:idx2_N-1]).*u[idx2_1+1:idx2_N+1] - ((ϵ2-1)/λ^2).*u[idx2_1-1:idx2_N-1].*u[idx2_1-2:idx2_N-2]
    Γ_v_2     = conj(u[idx1_1+1:idx1_N+1]).*u[idx2_1+2:idx2_N+2] .- (ϵv/λ).*conj(u[idx1_1-1:idx1_N-1]).*u[idx1_1+1:idx1_N+1] - ((ϵv-1)/λ^2).*u[idx1_1-1:idx1_N-1].*u[idx2_1-2:idx2_N-2]
    Γ_var1_2  = conj(u[idx2_1+1:idx2_N+1]).*u[idx1_1+2:idx1_N+2] .- (ϵvar1/λ).*conj(u[idx2_1-1:idx2_N-1]).*u[idx1_1+1:idx1_N+1] - ((ϵv-1)/λ^2).*u[idx1_1-1:idx1_N-1].*u[idx1_1-2:idx1_N-2]
    Γ_var2_2  = conj(u[idx1_1+1:idx1_N+1]).*u[idx1_1+2:idx1_N+2] .- (ϵvar2/λ).*conj(u[idx1_1-1:idx1_N-1]).*u[idx2_1+1:idx2_N+1] - ((ϵvar2-1)/λ^2).*u[idx2_1-1:idx2_N-1].*u[idx1_1-2:idx1_N-2]
    Γ_box_2   = ϵul.*u[idx1_1:idx1_N].*conj(u[idx1_1+1:idx1_N+1]) .+ (ϵur/λ).*u[idx1_1:idx1_N].*conj(u[idx1_1-1:idx1_N-1]) .+ ϵll.*u[idx1_1:idx1_N].*u[idx2_1+1:idx2_N+1] .+(ϵlr/λ).*u[idx1_1:idx1_N].*u[idx2_1-1:idx2_N-1] 

    @. du[idx1_1:idx1_N] = im*kn[idx1_1:idx1_N]*(Γ_sabra_1 + Γ_v_1  + Γ_var1_1 + Γ_var2_1 + Γ_box_1)-ν*kn[idx1_1:idx1_N]^(-2) *u[idx1_1:idx1_N] + fn - ν*kn[idx1_1:idx1_N]^(2)*u[idx1_1:idx1_N]
    @. du[idx2_1:idx2_N] = im*kn[idx1_1:idx1_N]*(Γ_sabra_2 + Γ_v_2  + Γ_var1_2 + Γ_var2_2 + Γ_box_2)-ν*kn[idx1_1:idx1_N]^(-2) *u[idx2_1:idx2_N] + fn - ν*kn[idx1_1:idx1_N]^(2)*u[idx2_1:idx2_N]
    return du
end 

dt = 5e-5
tspan = (0.0,120)
prob = ODEProblem(Sabra_coupled!, u0, tspan, p)
println("Solving...")
#sol = 0
begin @time sol = solve(prob, Rodas5P(autodiff = false),saveat = 1e-4,alg_hints = [:stiff],maxiters = 5e8,dt = dt, adaptive= false, progress = true) end


num_steps = size(sol, 2)
no_points = Int(round(num_steps*10^-3))
# plot the time series solutions of the shells
# Create a color gradient based on the number of time steps
colors = cgrad(:viridis, N, categorical=true)

# initial plots with the first column
p1 = plot(sol.t[1:no_points:end],real(sol[idx1_1, 1:no_points:end]), label="u[3]", title="Time series of Re(u) and Im(u)",ylabel="Re(u_1)",color=colors[1],legend=:none)#, ylims = [-0.05,0.05])
p2 = plot(sol.t[1:no_points:end],imag(sol[idx1_1, 1:no_points:end]), label="u[3]", ylabel="Im(u_1)",colorbar=true,color=colors[1],legend=:none)#,ylims = [-0.05,0.05])
p3 = plot(sol.t[1:no_points:end],real(sol[idx2_1, 1:no_points:end]), label="u[3]" ,ylabel="Re(u_2)",color=colors[1],legend=:none)#,ylims = [-0.05,0.05])
p4 = plot(sol.t[1:no_points:end],imag(sol[idx2_1, 1:no_points:end]),xlabel = "time", label="u[3]", ylabel="Im(u_2)",colorbar=true,color=colors[1],legend=:none)#,ylims = [-0.05,0.05])

# Now add rest of the plots
for i in 1:N
    plot!(p1, sol.t[1:no_points:end],real(sol[idx1_1+i, 1:no_points:end]), label="u[$i]",color=colors[i],legend=false)
    plot!(p2, sol.t[1:no_points:end],imag(sol[idx1_1+i, 1:no_points:end]), label="u[$i]",color=colors[i],legend=false)
    plot!(p3, sol.t[1:no_points:end],real(sol[idx2_1+i, 1:no_points:end]), label="u[$i]",color=colors[i],legend=false)
    plot!(p4, sol.t[1:no_points:end],imag(sol[idx2_1+i, 1:no_points:end]), label="u[$i]",color=colors[i],legend=false)
end

# Make a layout for the plots with the colorbar for the shell number 
l = @layout[grid(4,1) a{0.04w}] 
time_series = plot(p1, p2, p3, p4,heatmap((1:N).*ones(N,1),title="n",c=colors, legend=:none, xticks=:none, yticks=(1:0.1*N:N*1.1, string.(0:0.1*N:N))), layout = l) # display the plot

u_abs_1 =  mean(abs.(sol[idx1_1:idx1_N,1:end]), dims = 2) 
u_abs_2 =  mean(abs.(sol[idx2_1:idx2_N,1:end]), dims = 2) 

u0_abs_1 = mean(abs.(u0¹),dims = 2)
u0_abs_2 = mean(abs.(u0²),dims = 2)

plot(log10.(kn[idx1_1:idx1_N]),log10.(u_abs_1), label = L"$⟨|u|_{1,t>T/2}⟩$ ", xlabel = L"log_{10}(k)",ylabel = L"log_{10}(|u|)",lc = :black, dpi = 400)
plot!(title = L"$k^2$, $k^{-2}$ and $f_i = \delta_{i = [10,11]}$")
xticks!(log10.(kn[idx1_1:1:idx1_N]),string.(1:1:20))
plot!(log10.(kn[idx1_1:idx1_N]),log10.(u_abs_2), label = L"$⟨|u|_{2,t>T/2}⟩$ ", ls = :dash,lc = :black, dpi = 400)
plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1/3)), ylim = [-8,0], label = L"k^{-1/3}", ls = :dash, lc = :black)
plot!([log10(kn[7+2])], seriestype = "vline", label = L"\eta", lc = :black, ls = :dot)
plot!([log10(kn[8+2])], seriestype = "vline", label = L"\eta", lc = :black, ls = :dot)
plot!(log10.(kn[idx1_1:idx1_N]),log10.(u0_abs_1), label = L"k^{-1/3}", ls = :dot, lc = :black)
plot!(log10.(kn[idx1_1:idx1_N]),log10.(u0_abs_2), label = L"k^{-1/3}", ls = :dot, lc = :black)
#plot!([log10(k_η)], seriestype = "vline", label = L"\eta", lc = :black, ls = :dot)
#savefig(pwd()*"/u_vs_k_eq.png")

#jldsave(pwd()*"/3d2d_mix5_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2", sol = sol)