# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

# # Decaying Homogeneous Isotropic Turbulence - 2D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

# We start by loading packages.
# A [Makie](https://github.com/JuliaPlots/Makie.jl) plotting backend is needed
# for plotting. `GLMakie` creates an interactive window (useful for real-time
# plotting), but does not work when building this example on GitHub.
# `CairoMakie` makes high-quality static vector-graphics plots.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
name = "DecayingTurbulence2D"

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Viscosity model
Re = T(10_000)

# A 2D grid is a Cartesian product of two vectors
n = 256
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
# plot_grid(x...)

# Build setup and assemble operators
setup = Setup(x...; Re, ArrayType);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(setup);

u₀, p₀ = random_field(setup, T(0); pressure_solver);

# Solve unsteady problem
u, p, outputs = solve_unsteady(
    setup,
    u₀,
    p₀,
    (T(0), T(1));
    Δt = T(1e-3),
    pressure_solver,
    inplace = true,
    processors = (
        field_plotter(setup; nupdate = 20),
        energy_history_plotter(setup; nupdate = 20, displayfig = false),
        energy_spectrum_plotter(setup; nupdate = 20, displayfig = false),
        ## animator(setup, "vorticity.mp4"; nupdate = 16),
        ## vtk_writer(setup; nupdate = 10, dir = "output/$name", filename = "solution"),
        ## field_saver(setup; nupdate = 10),
        step_logger(; nupdate = 100),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

# Field plot
outputs[1]

# Energy history plot
outputs[2]

# Energy spectrum plot
outputs[3]

# Export to VTK
save_vtk(setup, u, p, "output/solution")

# Plot pressure
plot_pressure(setup, p₀)

# Plot velocity
plot_velocity(setup, u)

# Plot vorticity
plot_vorticity(setup, u)
