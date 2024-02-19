using Plots, JLD2, DifferentialEquations,LaTeXStrings,Statistics
#=
#Load simulated data
JLD2.@load "Data/3d2d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
sol_full = sol
JLD2.@load "Data/3d2d_mix1_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
sol_mix1 = sol
JLD2.@load "Data/3d2d_mix2_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
sol_mix2 = sol 
JLD2.@load "Data/2d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
sol2d = sol
JLD2.@load "Data/3d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
sol3d = sol
JLD2.@load "Data/3d_sabra_t200_f78.jld2" sol
sol3d_sabra = sol
JLD2.@load "Data/2d_sabra_t200_f78.jld2" sol
sol2d_sabra = sol
JLD2.@load "Data/3d_sabra_t200_f12.jld2" sol
sol3d_sabra_f12 = sol
=#

#General parameters for model simulated 
N = 20
λ = 2# lambda in k_n = k_0*lambda^n, canonically 2 for GOY and Sabra model
K0 = 1 # k_0 as above
kn = K0*λ.^(1:N) # kn's precomputed
kn = [0,0,kn...,0,0]
ν = 5e-7# Kinetic viscosity


#Indices for coupled model
idx1_1 = 3
idx1_N = N+2
idx2_1 = idx1_N+3
idx2_N = idx1_N + N + 2

#For Sabra model
idx_1 = idx1_1
idx_N = idx1_N


#Function calculating NL fluxes for sabra model
function NL_flux_sabra(u,ϵ)
    delta = mean(conj(u[idx_1-1:idx_N-1,:]).*conj(u[idx_1:idx_N,:]).*u[idx_1+1:idx_N+1,:], dims = 2)
    delta = [0,delta...,0]
    return kn[2:end-2].*imag(delta[2:end].-(ϵ-1)/λ.*delta[1:end-1])
end

#Function calculating NL enstrophy fluxes for sabra model
function NL_enstrophy_flux_sabra(u,ϵ)
    delta = mean(conj(u[idx_1-1:idx_N-1,:]).*conj(u[idx_1:idx_N,:]).*u[idx_1+1:idx_N+1,:], dims = 2)
    delta = [0,delta...,0]
    return (1/(1.25-1)).^(2:N+2).*kn[2:end-2].*imag(delta[2:end].-1/λ.*delta[1:end-1])
end

#Function plotting energy fluxes for sabra model simulations 
function plot_sabra_energy_flux()
    Π_3d_f12 = NL_flux_sabra(sol3d_sabra_f12, 1/2)
    Π_3d = NL_flux_sabra(sol3d_sabra, 1/2)
    Π_2d = NL_flux_sabra(sol2d_sabra, 5/4)

    plot(Π_3d_f12, xlabel = L"Shell number $n$", ylabel = L"$⟨\Pi_n^{e}⟩ = \tilde{ε}$", label = L"3D: $f_i = f\delta_{i,(1,2)}$ ", dpi = 400, lc = :black)
    plot!(Π_3d,dpi = 400, label = L"3D: $f_i = f\delta_{i,(7,8)}$ ", lc = :black, ls = :dash)
    plot!(Π_2d, dpi = 400, label = L"2D: $f_i = f\delta_{i,(7,8)}$ ", lc = :black, ls = :dot)
    xticks!((1:1:20),string.(1:1:20))
    plot!(Π_2d, inset = bbox(0.75,0.4,0.2,0.2), subplot = 2, lc = :black, ls = :dot, primary = false, xlim=[8,13], ylim = [-0.0005,0.0005], yticks = [0])
    savefig(pwd()*"/Plots/Sabra_energy_flux.png")
end
plot_sabra_energy_flux()

#Function plotting enstropjhy flux of 2d sabra
function plot_sabra_enstrophy()
    Π= NL_enstrophy_flux_sabra(sol2d_sabra,1.25)
    p = sign.(Π).*log10.(abs.(Π).+1)
    plot(p, xlabel = L"Shell number $n$", ylabel = L"$⟨\Pi_n^{z}⟩$", label = L"2D: $f_i = f\delta_{i,(7,8)}$", dpi = 400, lc = :black)
    yticks!((-1:1:4), [L"$-10^{1}$",L"$0$",L"$10^{1}$",L"$10^{2}$",L"$10^{3}$",L"$10^{4}$"])
    xticks!((1:1:20),string.(1:1:20))
    savefig(pwd()*"/Plots/Sabra_enstrophy_flux.png")
end
plot_sabra_enstrophy()

#Function producing spectra plot for sabra simulations in 2d and 3d cases
function plot_sabra_spectra()
    JLD2.@load "Data/3d_sabra_t200_f78.jld2" sol
    sol3d_sabra = sol
    JLD2.@load "Data/2d_sabra_t400_f78.jld2" sol
    sol2d_sabra = sol
    JLD2.@load "Data/3d_sabra_t200_f12.jld2" sol
    sol3d_sabra_f12 = sol

    Π = NL_flux_sabra(sol3d_sabra_f12, 1/2)

    ε = mean(Π[5:10]) #Index by eye

    #Get η from eq. (1.11) - doesnt seem to depend on ε since ν dominates
    η = ν^(3/4)/(ε^(1/4))

    #Inverse is wavenumber almost just kn[1]^(-1) = 87 as in last sentance p. 73...
    k_η= η^(-1)

    u_abs_3dsabra_f12 =  mean(abs.(sol3d_sabra_f12[idx_1:idx_N,500001:end]), dims = 2) 
    u_abs_3dsabra =  mean(abs.(sol3d_sabra[idx_1:idx_N,500001:end]), dims = 2) 
    u_abs_2dsabra =  mean(abs.(sol2d_sabra[idx_1:idx_N,500001:end]), dims = 2) 
    
    sol3d_sabra = 0
    sol2d_sabra = 0
    sol3d_sabra_f12 = 0 
    GC.gc()

    plot(log10.(kn[idx_1:idx_N]),log10.(u_abs_3dsabra_f12), label = L"3D: $f_i = f\delta_{i,(1,2)}$ ", xlabel = L"Shell number $n$",ylabel = L"log_{10}(|u|)",lc = :black, ylim = [-10,1],dpi = 400)
    xticks!(log10.(kn[idx_1:1:idx_N]),string.(1:1:20))
    plot!(log10.(kn[idx_1:idx_N]),log10.(u_abs_3dsabra), label = L"3D: $f_i = f\delta_{i,(7,8)}$ ", lc = :blue, dpi = 400)
    plot!(log10.(kn[idx_1:idx_N]),log10.(u_abs_2dsabra), label = L"2D: $f_i = f\delta_{i,(7,8)}$ ", lc = :red, dpi = 400)

    plot!(log10.(kn[idx_1:idx_N]),log10.(kn[idx_1:idx_N].^(-1/3)).+.5, label = L"k^{-1/3}", ls = :dash, lc = :black)
    plot!(log10.(kn[idx_1:idx_N]),log10.(kn[idx_1:idx_N].^(-1)).+1.5, label = L"k^{-1}", ls = :dot, lc = :black)

    plot!([log10(k_η)], seriestype = "vline", label = L"\eta", lc = :black, ls = :dashdot, primary = false)

    annotate!(4.8,-9.5,text(L"$k_{\eta}$"))

    savefig(pwd()*"/Plots/Sabra_spectrum.png")
end
plot_sabra_spectra()

#Function plotting f = ν = 0 equipartition in full interaction model
function plot_equipartition()
    JLD2.@load "Data/3d2d_t1_dt5e7_k01_L2_nu0.jld2" sol
    sol_equi_1 = sol
    JLD2.@load "Data/3d2d_t1-2_dt5e7_k01_L2_nu0.jld2" sol
    sol_equi_2 = sol
    abs_vals1 = hcat(abs.(sol_equi_1[idx1_1:idx1_N,:]),abs.(sol_equi_2[idx1_1:idx1_N,:]))
    u_abs_equi1 = mean(abs_vals1[:,10000:end], dims = 2)

    abs_vals2 = hcat(abs.(sol_equi_1[idx2_1:idx2_N,:]),abs.(sol_equi_2[idx2_1:idx2_N,:]))
    u_abs_equi2 = mean(abs_vals2[:,10000:end], dims = 2)
    
    u0_abs1 = abs.(sol_equi_1[idx1_1:idx1_N,1])
    u0_abs2 = abs.(sol_equi_1[idx2_1:idx2_N,1])

    sol_equi_1 = 0
    sol_equi_2 = 0
    GC.gc()


    plot(log10.(kn[idx_1:idx_N]),log10.(u_abs_equi1), label = L"⟨|u^{(1)}|⟩_{t>1}", xlabel = L"Shell number $n$",ylabel = L"log_{10}(|u|)",lc = :black,dpi = 400)
    plot!(log10.(kn[idx_1:idx_N]),log10.(u_abs_equi2), label = L"⟨|u^{(2)}|⟩_{t>1}",lc = :black, ls = :dot,dpi = 400)
    xticks!(log10.(kn[idx_1:1:idx_N]),string.(1:1:20))
    plot!(log10.(kn[idx_1:idx_N]),log10.(u0_abs1), label = L"$|u_{0}^{(1)}|$",lc = :red,dpi = 400)
    plot!(log10.(kn[idx_1:idx_N]),log10.(u0_abs2), label = L"$|u_{0}^{(2)}|$",lc = :red, ls = :dot,dpi = 400)
    

    savefig(pwd()*"/Plots/Equipartition.png")
end
plot_equipartition()

#Function plotting spectra of coupled model
function plot_spectrum()
    #plotting ro t = 20 (2*) to 120 (12*)
    tt = 5*100000
    #Load data, do computation, assign variable and delete and garbage collect
    
    #=
    JLD2.@load "Data/3d2d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol_full = sol
    uf_abs_1 =  mean(abs.(sol_full[idx1_1:idx1_N,tt:end]), dims = 2) 
    uf_abs_2 =  mean(abs.(sol_full[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol_full = nothing
    GC.gc()
    =#

    JLD2.@load "Data/3d2d_mix1_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol_mix1 = sol
    umix1_abs_1 =  mean(abs.(sol_mix1[idx1_1:idx1_N,tt:end]), dims = 2) 
    umix1_abs_2 =  mean(abs.(sol_mix1[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol_mix1 = nothing
    GC.gc()

    JLD2.@load "Data/3d2d_mix2_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol_mix2 = sol 
    umix2_abs_1 =  mean(abs.(sol_mix2[idx1_1:idx1_N,tt:end]), dims = 2) 
    umix2_abs_2 =  mean(abs.(sol_mix2[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol_mix2 = nothing
    GC.gc()

    JLD2.@load "Data/3d2d_mix3_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol_mix3 = sol 
    umix3_abs_1 =  mean(abs.(sol_mix3[idx1_1:idx1_N,tt:end]), dims = 2) 
    umix3_abs_2 =  mean(abs.(sol_mix3[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol_mix3 = nothing
    GC.gc()

    JLD2.@load "Data/3d2d_mix4_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol_mix4 = sol 
    umix4_abs_1 =  mean(abs.(sol_mix4[idx1_1:idx1_N,tt:end]), dims = 2) 
    umix4_abs_2 =  mean(abs.(sol_mix4[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol_mix4 = nothing
    GC.gc()

    #JLD2.@load "Data/3d2d_mix5_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    #sol_mix5 = sol 
    #umix5_abs_1 =  mean(abs.(sol_mix5[idx1_1:idx1_N,tt:end]), dims = 2) 
    #umix5_abs_2 =  mean(abs.(sol_mix5[idx2_1:idx2_N,tt:end]), dims = 2) 
    #sol_mix5 = nothing
    #GC.gc()

    JLD2.@load "Data/2d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol2d = sol
    u2d_abs_1 =  mean(abs.(sol2d[idx1_1:idx1_N,tt:end]), dims = 2) 
    u2d_abs_2 =  mean(abs.(sol2d[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol2d = nothing
    GC.gc()

    JLD2.@load "Data/3d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
    sol3d = sol
    u3d_abs_1 =  mean(abs.(sol3d[idx1_1:idx1_N,tt:end]), dims = 2) 
    u3d_abs_2 =  mean(abs.(sol3d[idx2_1:idx2_N,tt:end]), dims = 2) 
    sol3d = nothing
    GC.gc()

    plot(log10.(kn[idx1_1:idx1_N]),log10.(umix1_abs_1), label = "Mix 1", xlabel = L"Shell number $n$",ylabel = L"log_{10}\left(\left|u^{(1)}\right|\right)",lc = :red, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix2_abs_1), label = "Mix 2",lc = :blue, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix3_abs_1), label = "Mix 3",lc = :pink, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix4_abs_1), label = "Mix 4",lc = :green, dpi = 400)
    #plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix5_abs_1), label = "Mix 5",lc = :cyan, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(u2d_abs_1), label = "2D",lc = :black, dpi = 400)    
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(u3d_abs_1), label = "3D",lc = :black, dpi = 400, ls = :dash)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1/3)), ylim = [-8,0], label = L"k^{-1/3}", ls = :dashdot, lc = :black)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1)).+2, ylim = [-8,0], label = L"k^{-1}", ls = :dot, lc = :black)
    xticks!(log10.(kn[idx_1:1:idx_N]),string.(1:1:20))
    savefig(pwd()*"/Plots/Spectrum_v1.png")

    plot(log10.(kn[idx1_1:idx1_N]),log10.(umix1_abs_2), label = "Mix 1", xlabel = L"Shell number $n$",ylabel = L"log_{10}\left(\left|u^{(2)}\right|\right)",lc = :red, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix2_abs_2), label = "Mix 2",lc = :blue, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix3_abs_2), label = "Mix 3",lc = :pink, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix4_abs_2), label = "Mix 4",lc = :green, dpi = 400)
    #plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix5_abs_2), label = "Mix 5",lc = :cyan, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(u2d_abs_2), label = "2D",lc = :black, dpi = 400)    
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(u3d_abs_2), label = "3D",lc = :black, dpi = 400, ls = :dash)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1/3)), ylim = [-8,0], label = L"k^{-1/3}", ls = :dashdot, lc = :black)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1)).+2, ylim = [-8,0], label = L"k^{-1}", ls = :dot, lc = :black)
    xticks!(log10.(kn[idx_1:1:idx_N]),string.(1:1:20))
    savefig(pwd()*"/Plots/Spectrum_v2.png")
end
plot_spectrum()

#functiuon plotting energy of coupled model
function plot_energy()
       #plotting ro t = 20 (2*) to 120 (12*)
       tt = 5*100000
       #Load data, do computation, assign variable and delete and garbage collect
       
       #=
       JLD2.@load "Data/3d2d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol_full = sol
       uf_abs_1 =  mean(abs.(sol_full[idx1_1:idx1_N,tt:end]), dims = 2) 
       uf_abs_2 =  mean(abs.(sol_full[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol_full = nothing
       GC.gc()
       =#
   
       JLD2.@load "Data/3d2d_mix1_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol_mix1 = sol
       umix1_abs_1 =  mean(abs2.(sol_mix1[idx1_1:idx1_N,tt:end]), dims = 2) 
       umix1_abs_2 =  mean(abs2.(sol_mix1[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol_mix1 = nothing
       GC.gc()
   
       JLD2.@load "Data/3d2d_mix2_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol_mix2 = sol 
       umix2_abs_1 =  mean(abs2.(sol_mix2[idx1_1:idx1_N,tt:end]), dims = 2) 
       umix2_abs_2 =  mean(abs2.(sol_mix2[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol_mix2 = nothing
       GC.gc()
   
       JLD2.@load "Data/3d2d_mix3_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol_mix3 = sol 
       umix3_abs_1 =  mean(abs2.(sol_mix3[idx1_1:idx1_N,tt:end]), dims = 2) 
       umix3_abs_2 =  mean(abs2.(sol_mix3[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol_mix3 = nothing
       GC.gc()
   
       JLD2.@load "Data/3d2d_mix4_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol_mix4 = sol 
       umix4_abs_1 =  mean(abs2.(sol_mix4[idx1_1:idx1_N,tt:end]), dims = 2) 
       umix4_abs_2 =  mean(abs2.(sol_mix4[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol_mix4 = nothing
       GC.gc()
   
       JLD2.@load "Data/3d2d_mix5_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol_mix5 = sol 
       umix5_abs_1 =  mean(abs2.(sol_mix5[idx1_1:idx1_N,tt:end]), dims = 2) 
       umix5_abs_2 =  mean(abs2.(sol_mix5[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol_mix5 = nothing
       GC.gc()
   
       JLD2.@load "Data/2d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol2d = sol
       u2d_abs_1 =  mean(abs2.(sol2d[idx1_1:idx1_N,tt:end]), dims = 2) 
       u2d_abs_2 =  mean(abs2.(sol2d[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol2d = nothing
       GC.gc()
   
       JLD2.@load "Data/3d_t120_dt5e5_f78_k2_k-2_k01_L2_nu5e7.jld2" sol
       sol3d = sol
       u3d_abs_1 =  mean(abs2.(sol3d[idx1_1:idx1_N,tt:end]), dims = 2) 
       u3d_abs_2 =  mean(abs2.(sol3d[idx2_1:idx2_N,tt:end]), dims = 2) 
       sol3d = nothing
       GC.gc()
   
       plot(log10.(kn[idx1_1:idx1_N]),log10.(umix1_abs_1), label = "Mix 1", xlabel = L"Shell number $n$",ylabel = L"log_{10}(|u|)",lc = :red, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix2_abs_1), label = "Mix 2",lc = :blue, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix3_abs_1), label = "Mix 3",lc = :pink, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix4_abs_1), label = "Mix 4",lc = :green, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix5_abs_1), label = "Mix 5",lc = :cyan, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(u2d_abs_1), label = "2D",lc = :black, dpi = 400)    
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(u3d_abs_1), label = "3D",lc = :black, dpi = 400, ls = :dash)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-3)), label = L"k^{-3}", ls = :dot, lc = :black)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-5/3)), label = L"k^{-5/3}", ls = :dashdot, lc = :black)
       xticks!(log10.(kn[idx_1:1:idx_N]),string.(1:1:20))
       savefig(pwd()*"/Plots/Energy_Spectrum_v1.png")
       #=
       plot(log10.(kn[idx1_1:idx1_N]),log10.(umix1_abs_2), label = "Mix 1", xlabel = L"Shell number $n$",ylabel = L"log_{10}(|u|)",lc = :red, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix2_abs_2), label = "Mix 2",lc = :blue, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix3_abs_2), label = "Mix 3",lc = :pink, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix4_abs_2), label = "Mix 4",lc = :green, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(umix5_abs_2), label = "Mix 5",lc = :cyan, dpi = 400)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(u2d_abs_2), label = "2D",lc = :black, dpi = 400)    
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(u3d_abs_2), label = "3D",lc = :black, dpi = 400, ls = :dash)
       plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1/3)), ylim = [-8,0], label = L"k^{-1/3}", ls = :dot, lc = :black)
       xticks!(log10.(kn[idx_1:1:idx_N]),string.(1:1:20))
       savefig(pwd()*"/Plots/Spectrum_v2.png")
       =#
    
end
plot_energy()


function plot_ts(sol)
    num_steps = size(sol, 2)
    no_points = Int(round(num_steps*10^-3))
    # plot the time series solutions of the shells
    # Create a color gradient based on the number of time steps
    colors = cgrad(:viridis, 20, categorical=true)

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
    l = @layout[grid(4,1) a{0.04w}] 
    time_series = plot(p1, p2, p3, p4,heatmap((1:N).*ones(N,1),title="n",c=colors, legend=:none, xticks=:none, yticks=(1:0.1*N:N*1.1, string.(0:0.1*N:N))), layout = l) # display the plot
end

function plot_u(sol)
    u_abs_1 =  mean(abs.(sol[idx1_1:idx1_N,10000]), dims = 2) 
    u_abs_2 =  mean(abs.(sol[idx2_1:idx2_N,:]), dims = 2) 

    plot(log10.(kn[idx1_1:idx1_N]),log10.(u_abs_1), label = L"$⟨|u|_{1,t>T/2}⟩$ ", xlabel = L"log_{10}(k)",ylabel = L"log_{10}(|u|)",lc = :black, dpi = 400)
    plot!(title = L"$k^2$, $k^{-2}$ and $f_i = \delta_{i = [10,11]}$")
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(u_abs_2), label = L"$⟨|u|_{2,t>T/2}⟩$ ", ls = :dash,lc = :black, dpi = 400)
    plot!(log10.(kn[idx1_1:idx1_N]),log10.(kn[idx1_1:idx1_N].^(-1/3)), ylim = [-10,2], label = L"k^{-1/3}", ls = :dot, lc = :black)
end

#plot_spectrum(sol_full,sol2d,sol3d)
#plot_ts(sol2d)

#Compute first invariant

#Make function that computes alphas for second invariant, and then computes 
#second conserved integral for the time series

#Plot the time series using Samuels beautiful plotting code

#Plot spectra of [u_n^(1),u_n^(2)] and seperates, and kolmogorv scaling.

#Plot time series of the conserved integrals. Mostly interesting the in unforced and non-dissipative case.
function energy(sol)
    E1  = transpose(sum(abs2.(sol[idx1_1:idx1_N,1:end]), dims = 1))
    E2  = transpose(sum(abs2.(sol[idx2_1:idx2_N,1:end]), dims = 1))
    return E1, E2
end

function alphas(ϵ, λ)
    return (log10(1/(ϵ-1))+im*pi)/log(λ)
    
end

function second_conserved(sol)
    ϵ1 = 0.5
    ϵ2 = 1.25
    ϵv = 0.8
    ϵvar1 = 0.8
    ϵvar2 = 1.2
    ϵul = 0.8
    ϵur = 0.8
    ϵll = 1.2
    ϵlr = 1.2
    
    α1 = alphas(ϵ1,λ)
    α2 = alphas(ϵ2,λ)
    αv = alphas(ϵv,λ)
    αvar1 = alphas(ϵvar1,λ)
    αvar2 = alphas(ϵvar2,λ)
    αul = alphas(ϵul,λ)
    αur = alphas(ϵur,λ)
    αll = alphas(ϵll,λ)
    αlr = alphas(ϵlr,λ)

    

end

