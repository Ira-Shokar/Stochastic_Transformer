# coding=utf-8
# Copyright (c) 2023 Ira Shokar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on the GeophysicalFlows example

using Pkg
Pkg.activate("/home/is500/Documents/QGF")

using GeophysicalFlows, CUDA, Random, Printf, JLD2 #, Metal
using LinearAlgebra: ldiv!
import FourierFlows as FF

### HELPER FUNCTIONS ##########################################################################

function load_qh(load_seed, β, load_timestep)
    ### LOAD PREVIOUS FILE ###
    load_file = jldopen("/home/is500/Documents/QGF/outputs/singlelayerqg_forcedbeta_$(load_seed)_$(β).jld2", "r")
    step      = parse.(Int, keys(load_file["snapshots/t"]))[load_timestep]
    qh_load   = load_file["snapshots/qh/$step"]
    close(load_file)
    return qh_load
end

get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution

function get_u(prob)
  @. prob.vars.qh = prob.sol

  SingleLayerQG.streamfunctionfrompv!(prob.vars.ψh, prob.vars.qh, prob.params, prob.grid)

  ldiv!(prob.vars.u, prob.grid.rfftplan, -im * prob.grid.l .* prob.vars.ψh)

  return Array(prob.vars.u)
end

function forcingspectrum(ε, k_f, k_width, grid::AbstractGrid)

    K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber
    
    forcing_spectrum = @. exp(-(K - k_f)^2 / (2 * k_width^2))
    @CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average
    
    ε0 = FF.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
    @. forcing_spectrum *= ε/ε0;       # normalize forcing to inject energy at rate ε
    return forcing_spectrum
end

function calcF!(Fh, sol, t, clock, vars, params, grid)
    T = eltype(grid)
    @. Fh = sqrt(forcing_spectrum) * cis(2π * rand(T)) / sqrt(clock.dt)
    return nothing
end;

function set_qh!(prob, qh)

    @. prob.vars.qh = qh 
    @. prob.sol = prob.vars.qh
  
    SingleLayerQG.updatevars!(prob.sol, prob.vars, prob.params, prob.grid)
  
    return nothing
  end

### MAIN FUNCTION ##########################################################################

function main(seed)

    println(seed)

    startwalltime = time()

    if dev==CPU(); Random.seed!(seed); else; CUDA.seed!(seed); end;

    prob = SingleLayerQG.Problem(
        dev; nx=n,ny=n, Lx=L, Ly = L, β=β, μ=μ, ν=ν, nν = nν, dt,
        stepper="FilteredRK4", calcF=calcF!, stochastic=true, aliased_fraction = 1/3
        );

    set_qh!(prob, device_array(dev)(qh_load));

    E = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
    Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
    diags = [E, Z]; # A list of Diagnostics types passed to "stepforward!" will be updated every timestep.

    filename = "outputs/singlelayerqg_forcedbeta_$(seed)_$(β).jld2";

    if isfile(filename); rm(filename); end

    output = Output(prob, filename, (:qh, get_sol), (:u, get_u))

    saveproblem(output)
    saveoutput(output)

    while prob.clock.step <= nsteps

        stepforward!(prob, diags, save_substeps)
        SingleLayerQG.updatevars!(prob)

        if prob.clock.step % save_substeps == 0
            saveoutput(output)

            println(@sprintf("seed: %04d, step: %04d, t: %d, E: %.3e, Q: %.3e, walltime: %.2f min",
            seed, prob.clock.step, prob.clock.t, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)
            )
            
        end
    end

    savediagnostic(E, "energy"   , output.path)
    savediagnostic(Z, "enstrophy", output.path)
end

### RUN MAIN IN PARALLEL ##########################################################################

          dev = GPU();  # Device (CPU/GPU)

            L = 2π                    # domain size
            n = 256                   # 2D resolution: n² grid points

            μ = 4e-4                  # bottom drag

           nν = 8                     # hyperviscosity order
            ν = (n/3)^(-nν*2)         # hyperviscosity coefficent

           dt = 4e-2                  # timestep
        t_max = 200                   # integration time
save_substeps = 2500                  # number of timesteps after which output is saved
       nsteps = t_max*save_substeps   # total number of timesteps

      ε = 1e-6          # forcing energy input rate
    k_f = 16.0 * 2π/L   # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
k_width = 1.5  * 2π/L   # the width of the forcing spectrum, `δ_f`

β = 1.35 # planetary PV gradient
load_seed = 20
load_timestep = 844

qh_load = load_qh(load_seed, β, load_timestep)
forcing_spectrum = forcingspectrum(ε, k_f, k_width, FF.TwoDGrid(dev; nx=n, Lx=L))


for s in 5000:6000
    main(s)
end