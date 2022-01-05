"""
Construct divergence and gradient operator
"""
function operator_divergence!(setup)
    @unpack bc, model = setup
    @unpack problem = setup.case
    @unpack pressure_solver = setup.solver_settings
    @unpack Nx, Ny, Nz = setup.grid
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t, Nuz_in, Nuz_b, Nuz_t = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t, Nvz_in, Nvz_b, Nvz_t = setup.grid
    @unpack Nwx_in, Nwx_b, Nwx_t, Nwy_in, Nwy_b, Nwy_t, Nwz_in, Nwz_b, Nwz_t = setup.grid
    @unpack Nu, Nv, Nw, Np, Npx, Npy, Npz = setup.grid
    @unpack hx, hy, hz = setup.grid
    @unpack Ω⁻¹ = setup.grid

    ## Divergence operator M

    # Note that the divergence matrix M is not square
    mat_hx = spdiagm(hx)
    mat_hy = spdiagm(hy)
    mat_hz = spdiagm(hz)

    ## Mx
    # Building blocks consisting of diagonal matrices where the diagonal is
    # Equal to constant per block (hy(block)) and changing for next block to
    # Hy(block+1)
    diag1 = ones(Nux_t - 1)
    M1D = spdiagm(Nux_t - 1, Nux_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restrict)
    diagpos = 0
    if bc.u.x[2] == :pressure && bc.u.x[1] == :pressure
        diagpos = 1
    end
    if bc.u.x[2] != :pressure && bc.u.x[1] == :pressure
        diagpos = 1
    end
    if bc.u.x[2] == :pressure && bc.u.x[1] != :pressure
        diagpos = 0
    end
    if bc.u.x[2] == :periodic && bc.u.x[1] == :periodic
        # Like pressure left
        diagpos = 1
    end

    BMx = spdiagm(Npx, Nux_t - 1, diagpos => ones(Npx))
    M1D = BMx * M1D

    # Extension to 3D to be used in post-processing files
    Bup = I(Nuz_in) ⊗ I(Nuy_in) ⊗ BMx

    # Boundary conditions
    Mx_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])
    Mx_bc = (; Mx_bc..., Bbc = mat_hz ⊗ mat_hy ⊗ (M1D * Mx_bc.Btemp))

    # Extend to 3D
    Mx = mat_hz ⊗ mat_hy ⊗ (M1D * Mx_bc.B1D)

    ## My
    # Same as Mx but reversing indices and kron arguments
    diag1 = ones(Nvy_t - 1)
    M1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restriction)
    diagpos = 0
    if bc.v.y[2] == :pressure && bc.v.y[1] == :pressure
        diagpos = 1
    end
    if bc.v.y[2] != :pressure && bc.v.y[1] == :pressure
        diagpos = 1
    end
    if bc.v.y[2] == :pressure && bc.v.y[1] != :pressure
        diagpos = 0
    end
    if bc.v.y[2] == :periodic && bc.v.y[1] == :periodic
        # Like pressure low
        diagpos = 1
    end

    BMy = spdiagm(Npy, Nvy_t - 1, diagpos => ones(Npy))
    M1D = BMy * M1D

    # Extension to 3D to be used in post-processing files
    Bvp = I(Nvz_in) ⊗ BMy ⊗ I(Nvx_in)

    # Boundary conditions
    My_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])
    My_bc = (; My_bc..., Bbc = mat_hz ⊗ (M1D * My_bc.Btemp) ⊗ mat_hx)

    # Extend to 3D
    My = mat_hz ⊗ (M1D * My_bc.B1D) ⊗ mat_hx

    ## Mz
    # Same as Mx but reversing indices and kron arguments
    diag1 = ones(Nwz_t - 1)
    M1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => -diag1, 1 => diag1)

    # We only need derivative at inner pressure points, so we map the resulting
    # Boundary matrix (restriction)
    diagpos = 0
    bc.w.z[1] == :pressure && bc.w.z[2] == :pressure && (diagpos = 1)
    bc.w.z[1] == :pressure && bc.w.z[2] != :pressure && (diagpos = 1)
    bc.w.z[1] != :pressure && bc.w.z[2] == :pressure && (diagpos = 0)
    bc.w.z[1] == :periodic && bc.w.z[2] == :periodic && (diagpos = 1)

    BMz = spdiagm(Npz, Nwz_t - 1, diagpos => ones(Npz))
    M1D = BMz * M1D

    # Extension to 3D to be used in post-processing files
    Bwp = BMz ⊗ I(Nwy_in) ⊗ I(Nwx_in)

    # Boundary conditions
    Mz_bc = bc_general(Nwz_t, Nwz_in, Nwz_b, bc.w.z[1], bc.w.z[2], hz[1], hz[end])
    Mz_bc = (; Mz_bc..., Bbc = (M1D * Mz_bc.Btemp) ⊗ mat_hy ⊗ mat_hx)

    # Extend to 3D
    Mz = (M1D * Mz_bc.B1D) ⊗ mat_hy ⊗ mat_hx

    ## Resulting divergence matrix
    M = [Mx My Mz]

    ## Gradient operator G

    # Like in the continuous case, grad = -div^T
    # Note that this also holds for outflow boundary conditions, if the stress
    # on the ouflow boundary is properly taken into account in y_p (often this
    # stress will be zero)
    Gx = -Mx'
    Gy = -My'
    Gz = -Mz'

    G = [Gx; Gy; Gz]

    ## Store in setup structure
    @pack! setup.discretization = M, Mx, My, Mz, Mx_bc, My_bc, Mz_bc, G, Gx, Gy, Gz
    @pack! setup.discretization = Bup, Bvp, Bwp

    ## Pressure matrix for pressure correction method;
    # Also used to make initial data divergence free or compute additional poisson solve
    if !is_steady(problem) && !isa(model, KEpsilonModel)
        # Note that the matrix for the pressure is constant in time.
        # Only the right hand side vector changes, so the pressure matrix can be set up
        # outside the time-stepping-loop.

        # Laplace = div grad
        A = M * Diagonal(Ω⁻¹) * G
        @pack! setup.discretization = A

        # ROM does not require Poisson solve for simple BC
        # For rom_bc > 0, we need Poisson solve to determine the V_bc field
        if setup.rom.use_rom && setup.rom.rom_bc == 0 && setup.rom.rom_type == "POD"
            return setup
        end

        initialize!(pressure_solver, setup, A)

        # Check if all the row sums of the pressure matrix are zero, which
        # should be the case if there are no pressure boundary conditions
        if any(isequal(:pressure), [bc.u.x..., bc.v.y..., bc.w.z...])
            if any(!isapprox(0; atol = 1e-10), abs.(sum(A; dims = 2)))
                @warn "Pressure matrix: not all rowsums are zero!"
            end
        end
    end

    setup
end
