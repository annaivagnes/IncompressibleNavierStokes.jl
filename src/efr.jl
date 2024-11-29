"""
Differential filter for the incompressible Navier-Stokes equations.

## Exports

The following symbols are exported by Filter:

$(EXPORTS)
"""

using LinearAlgebra
using SparseArrays

export differential_filter

"""
    differential_filter(u, v, w)

Applies a Helmholtz differential filter to the velocity fields (u, v, w).
Returns the filtered velocity fields (u_filtered, v_filtered, w_filtered).
"""
function differential_filter(u, setup, filter_radius, relax_parameter, lu_filter_mats)
    # Extract grid and boundary information from the setup
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Nu, Ip, Iu, Δ, Δu) = grid
    D = dimension()
	T = Float64

#    # Apply filter across all velocity components (accounting for each dimension)
	ufilt = copy(u)
#
#    for α = 1:D
#		# old velocity (not filtered) only in internal points
#		u_internal = view(u[Iu[α], α], :)
#		# find new filtered velocity (when the diffusion matrix is applicable)
#		u_bc = view(diffusion_vec_bc(setup, u, α), :)
#		rhs = u_internal + 2*(filter_radius^2).*u_bc
#		ufilt_internal = lu_filter_mats[α] \ rhs
#		ufilt[Iu[α], α] .= reshape(ufilt_internal, size(ufilt[Iu[α], α]))
#    end
	D = diffusion_mat(setup)
	yu = apply_bc_u(zero(ufilt), 1.0, setup)
	B = bc_u_mat(setup)
	rhs = u[:] + 2*(filter_radius^2)*D*yu[:]
	ufilt .= reshape(lu_filter_mats \ rhs, size(u))

	# Optional relaxation step
	if !isnothing(relax_parameter)
		u_filt .= relax(u, ufilt, relax_parameter)
    end

    return u_filt
end

function differential_filter!(u, setup, filter_radius, relax_parameter, lu_filter_mats)
    # Extract grid and boundary information from the setup
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Nu, Ip, Iu, Δ, Δu) = grid
    D = dimension()
	T = Float64

	ufilt = copy(u)
	D = diffusion_mat(setup)
	yu = apply_bc_u(zero(ufilt), 1.0, setup)
	B = bc_u_mat(setup)
	rhs = u[:] + 2*(filter_radius^2)*D*yu[:]
	ufilt .= reshape(lu_filter_mats \ rhs, size(u))

	# Optional relaxation step
	if !isnothing(relax_parameter)
		relax!(u, ufilt, relax_parameter)
    end
    return u
end

function decompose_filter_mat(setup, filter_radius)
	(; dimension, Nu) = setup.grid
	D = dimension()
	D = diffusion_mat(setup)
	B = bc_u_mat(setup)
	Id = sparse(I, size(D))
	filter_mat = Id-2*(filter_radius^2)*D*B
	lu(filter_mat)
end

function relax(u, u_filtered, relax_parameter)
    u_relaxed = (1 - relax_parameter) .* u + relax_parameter .* u_filtered
    return u_relaxed
end

function relax!(u, u_filtered, relax_parameter)
    u .= (1 - relax_parameter) .* u + relax_parameter .* u_filtered
    return u
end
