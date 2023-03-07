# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pyvista

import dolfinx.fem as fem
import dolfinx.mesh as mesh
import dolfinx.io as io
import dolfinx.plot as plot
import ufl
import basix 

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import sys
from meshes import generate_mesh_with_crack, generate_cylinder, generate_hollow_cylinder_2D
from petsc_solvers import SNESSolver
from utils import ColorPrint

petsc_options_SNES = {
    "snes_type": "vinewtonrsls",
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1.0e-08,
    "snes_rtol": 1.0e-09,
    "snes_stol": 0.0,
    "snes_max_it": 50,
    "snes_monitor": "",
    # "snes_monitor_cancel": "",
}

petsc_options_SNESQN = {
    "snes_type": "qn",
    "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
    "snes_qn_m": 100,
    "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
    "snes_qn_restart_type": "none", #<powell,periodic,none> 
    "pc_type": "cholesky", # cholesky >> hypre > gamg,sor ; asm, lu, gas - don't work
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1.0e-08,
    "snes_rtol": 1.0e-08,
    "snes_stol": 0.0,
    "snes_max_it": 50,
    # "snes_monitor": "",
    "snes_monitor_cancel": "",
}


def solve_antiplane(
    R_i = 0.5,
    R_e = 1.,
    sig_0 = 1.,
    R = 0.1,
    lc =.1,
    dist_min = .1,
    dist_max = .3,
    refinement_ratio=1,
    results = True,
    results_name = "none",
    c_lim = 5.,
    t_max = 1.,
    incr = 15
    ):

        #------------------------------------------------------------------------
        # Mesh
        #------------------------------------------------------------------------
        msh, cell_tags, facet_tags = generate_hollow_cylinder_2D(
                R_i=R_i,
                R_e=R_e,
                lc=lc,  # caracteristic length of the mesh
                refinement_ratio=refinement_ratio,  # how much it is refined at the tip zone
                dist_min=dist_min,  # radius of tip zone
                dist_max=dist_max,  # radius of the transition zone
            )
        
        #------------------------------------------------------------------------
        # Finite element function space
        #------------------------------------------------------------------------
        deg_stress = 0 # constant on each element
        deg_u = 1 # linear on each element
        element = ufl.FiniteElement('Lagrange',msh.ufl_cell(),degree=deg_u)
        V = fem.FunctionSpace(msh, element) # displacement unknown is a scalar in antiplane elasticity : u = u(x,y)e_z
        element_stress = ufl.VectorElement("DG", msh.ufl_cell(), degree=deg_stress, quad_scheme='default')
        element_stress_scalar = ufl.FiniteElement("DG", msh.ufl_cell(), degree=deg_stress, quad_scheme='default') 
        V_stress = fem.FunctionSpace(msh, element_stress)
        V_stress_scalar = fem.FunctionSpace(msh, element_stress_scalar)

        #------------------------------------------------------------------------
        # Boundary conditions, integrals and normal
        #------------------------------------------------------------------------
        inner_facets = facet_tags.find(1)
        outer_facets = facet_tags.find(2)
        inner_dofs = fem.locate_dofs_topological(V, msh.topology.dim-1, inner_facets)
        outer_dofs = fem.locate_dofs_topological(V, msh.topology.dim-1, outer_facets)
        u_t = fem.Constant(msh, 1.) 
        bc_outer = fem.dirichletbc(np.array(0.,dtype=ScalarType), outer_dofs, V)
        bc_inner = fem.dirichletbc(u_t, inner_dofs, V)
        bcs = [bc_outer, bc_inner]
        dx = ufl.Measure("dx",domain=msh,  metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"} )
        ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
        n = ufl.FacetNormal(msh) # n is the normal vector of boundaries of msh

        #------------------------------------------------------------------------
        # Plastic criterion and linear form function
        #------------------------------------------------------------------------
        mu = fem.Constant(msh,PETSc.ScalarType(1.))
        def eps(u):
            """Strain"""
            return ufl.grad(u)  
        def L(v): 
            """The linear form of the weak formulation"""
            # Volume force
            b = fem.Constant(msh,ScalarType(0))
            # Surface force on the top 
            f = fem.Constant(msh,ScalarType(0.))
            return b * v * dx + f * v * ds(1)
        def norm_s(sig):
            return ufl.sqrt(3*ufl.inner(sig, sig))
        def sig_new(sigma_elas, sig_old, sig_0, p_old, R):  
            criterion = norm_s(sigma_elas)-sig_0-R*p_old
            direction = sigma_old/(norm_s(sigma_old))  
            return ufl.conditional(criterion > 0., sigma_elas - 3*mu/(R+3*mu)*criterion*direction, sigma_elas)
        def dp(sigma_elas, mu, sig_0, p_old, R):
            criterion = norm_s(sigma_elas)-sig_0-R*p_old
            return ufl.conditional(criterion > 0., criterion/(R+3.*mu), fem.Constant(msh,0.)) 
        def deps_p(sigma_elas, sigma, mu):
            criterion = sigma_elas-sigma 
            test = norm_s(criterion)
            return ufl.conditional( test > 0., criterion/mu, ufl.as_vector([0.,0.]))

        #------------------------------------------------------------------------
        # Initializing the problem
        #------------------------------------------------------------------------
        sig_0 = sig_0
        R = R
        # Function spaces
        u = fem.Function(V,name="u")
        u_old = fem.Function(V)
        v = ufl.TestFunction(V)       
        eps_p = fem.Function(V_stress,name="epsp")
        eps_p_old = fem.Function(V_stress)
        p = fem.Function(V_stress_scalar,name="p")
        p_old = fem.Function(V_stress_scalar)
        sigma = fem.Function(V_stress,name="stress")
        sigma_old = fem.Function(V_stress)
        # Function expressions
        sigma_elas = sigma_old+mu*eps(u-u_old)
        sigma_new = sig_new(sigma_elas, sigma_old, sig_0, p_old, R)
        stress_expr = fem.Expression(sigma_new, V_stress.element.interpolation_points())
        stress_elas_expr = fem.Expression(sigma_elas, V_stress.element.interpolation_points())
        p_expr = fem.Expression(p_old+dp(sigma_elas, mu, sig_0, p_old, R), V_stress_scalar.element.interpolation_points())
        eps_p_expr = fem.Expression(eps_p_old+deps_p(sigma_elas, sigma_new, mu), V_stress.element.interpolation_points())
        # Non linear problem
        residual = ufl.inner(sigma_new, eps(v)) * dx - L(v)
        my_problem = SNESSolver(residual, u, bcs=bcs, petsc_options=petsc_options_SNES)
        # Export to Paraview
        if not results_name == "none" :
            with io.XDMFFile(MPI.COMM_WORLD, "output/{}.xdmf".format(results_name), "w") as file:
                file.write_mesh(u.function_space.mesh)
        # Input and outputs
        loads = np.linspace(0,t_max,incr)
        plastic_dissipations = np.zeros_like(loads)
        Fi = []
        Fe = []
        
        #------------------------------------------------------------------------
        # Non linear solver
        #------------------------------------------------------------------------
        for i,t in enumerate(loads):
            u_old.x.array[:] = u.x.array
            p_old.x.array[:] = p.x.array
            eps_p_old.x.array[:] = eps_p.x.array
            sigma_old.x.array[:] = sigma.x.array
            # Apply the load
            u_t.value = t
            # Solve
                #ColorPrint.print_info(f"Solve for t={t:5.3f}")
            out = my_problem.solve()
                #ColorPrint.print_info(out)
            # Update 
            eps_p.interpolate(eps_p_expr)
            eps_p.x.scatter_forward()    
            p.interpolate(p_expr)
            p.x.scatter_forward()
            sigma.interpolate(stress_expr)
            sigma.x.scatter_forward()
            plastic_dissipations[i] = fem.assemble_scalar(fem.form(sig_0 * p * dx))
            # Compute boundary forces
            inner_force = fem.assemble_scalar(fem.form(ufl.dot(sigma, n)*ds(1)))
            outer_force = fem.assemble_scalar(fem.form(ufl.dot(sigma, n)*ds(2)))
            Fi.append(inner_force)
            Fe.append(outer_force)
            # Export to Paraview
            if not results_name == "none" :
                with io.XDMFFile(MPI.COMM_WORLD, "output/{}.xdmf".format(results_name), "a") as file:
                    file.write_function(u,t)
                    file.write_function(eps_p,t)
                    file.write_function(p,t)
                    file.write_function(sigma,t)  

        #------------------------------------------------------------------------
        # Print results
        #------------------------------------------------------------------------
        if results :
            pyvista.start_xvfb()
            # Create plotter and pyvista grid
            topology, cell_types, geometry = plot.create_vtk_mesh(V)
            u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            u_grid.cell_data["p"] = p.x.array.real
            u_grid.set_active_scalars("p")
            u_plotter = pyvista.Plotter(window_size=(800, 600))
            u_plotter.add_mesh(u_grid, show_edges=True,clim=[0,c_lim])
            u_plotter.view_xy()
            if not pyvista.OFF_SCREEN:
                u_plotter.show()
            print(plastic_dissipations)

        #------------------------------------------------------------------------
        # Returns useful data
        #------------------------------------------------------------------------
        return loads, Fi, Fe, plastic_dissipations
