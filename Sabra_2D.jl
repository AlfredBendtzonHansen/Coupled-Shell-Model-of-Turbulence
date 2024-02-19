#= 
Script for simualting and plotting spectra and mean dissipation energy for the Sabra model, 
    producing plots for the 2D regime.


=# 
using DifferentialEquations
using Plots
using LinearAlgebra
using Random
using DynamicalSystems
using LaTeXStrings
using Statistics
using Distributions
using JLD2
using FileIO
using ProgressLogging

N = 20#Number of shells

λ = 2# lambda in k_n = k_0*lambda^n, canonically 2 for GOY and Sabra model
K0 = 0.05 # k_0 as above
kn = K0*λ.^(1:N) # kn's precomputed

u0 = ones(ComplexF64,N).*kn.^(-1/3) # Initial u as complex array, set to zero

ν = 5e-7# Kinetic viscosity

fn = zeros(ComplexF64,N) #Forcing
fn[end-4:end-3] .= (1 + 1*im)*0.1# Only small wave nr (large scale) should force to obser cascading and intermittency

ϵ = 5/4

kn = [0,0,kn...,0,0]
u0 = [0,0,u0...,0,0]

idx_1 = 3
idx_N = N+2

p = tuple(kn, ϵ, fn, ν, idx_1, idx_N) #Collect parameters in tuple p to pass to function Sabra!(du, u, p , t)


function Sabra!(du, u, p, t)#change to actual typset vars
    kn,ε,fn,ν,idx_1,idx_N = p

    u1 = conj.(u[idx_1+1:idx_N+1]).*u[idx_1+2:idx_N+2]
    u2 = -(ε/λ).*conj.(u[idx_1-1:idx_N-1]).*u[idx_1+1:idx_N+1]
    u3 = -((ε-1.0)/λ^2).*u[idx_1-2:idx_N-2].*u[idx_1-1:idx_N-1]
    
    @. du[idx_1:idx_N] = im*kn[idx_1:idx_N]*(u1+u2+u3)-ν*kn[idx_1:idx_N]^(-2)*u[idx_1:idx_N]+fn
    return du
end 

dt = 1e-4
tspan = (0.0,100.0)
prob = ODEProblem(Sabra!, u0, tspan, p)
println("Solving...")
begin @time sol = solve(prob,alg_hints = [:stiff],maxiters = 5e8, dt = dt, adaptive= false, progress = true) end

function NL_flux(u)
    delta = mean(conj(u[idx_1-1:idx_N-1,:]).*conj(u[idx_1:idx_N,:]).*u[idx_1+1:idx_N+1,:], dims = 2)
    delta = [0,delta...,0,]
    return kn[2:end-2].*imag(delta[2:end].-(ϵ-1)/λ.*delta[1:end-1])
end

#Π = NL_flux(sol)

#Largest eddy-turnover time
t0_τ = (size(sol.t)[1]*dt)/(mean(abs.(sol[idx_1,:]))*kn[idx_1])^(-1)
plot_text = round(t0_τ,digits = 1)
#plot(Π,legend = false ,xlabel = L"Shell number $n$",ylabel = L"$⟨\Pi_n^{e}⟩ = \tilde{ε}$" , dpi = 400, lc = :black)
#plot!(title = "Mean dissipation energy (ν = $ν, t/τ₀ = $plot_text)")
#savefig(pwd()*"/dissipation_energy.png")

#ε = mean(Π[8:12])

#Get η from eq. (1.11) - doesnt seem to depend on ε since ν dominates
#η = ν^(3/4)/(ε^(1/4))

#Inverse is wavenumber almost just kn[1]^(-1) = 87 as in last sentance p. 73...
#k_η= η^(-1)

u_abs =  mean(abs.(sol[idx_1:idx_N,:]), dims = 2) 
plot(log10.(kn[idx_1:idx_N]),log10.(u_abs), label = L"$⟨|u|_{t>T/2}⟩$ ", xlabel = L"log_{10}(k)",ylabel = L"log_{10}(|u|)", title = "Spectral Velocity Scaling(ν = $ν, t/τ₀ = $plot_text)",lc = :black, dpi = 400)
plot!(log10.(kn[idx_1:idx_N]),log10.(kn[idx_1:idx_N].^(-1/3)).+.5, ylim = [-10,2], label = L"k^{-1/3}", ls = :dash, lc = :black)
#plot!([log10(k_η)], seriestype = "vline", label = L"\eta", lc = :black, ls = :dot)
savefig(pwd()*"/u_vs_k_sabra2D.png")
