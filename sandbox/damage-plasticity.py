import numpy as np
import yaml

from pathlib import Path
import sys
import os

from mpi4py import MPI
from petsc4py import PETSc
import basix

import argparse
import dolfinx
from dolfinx.io import XDMFFile
import ufl
import json

sys.path.append("../../damage")
from petsc_solvers import SNESSolver, TAOSolver
from utils import project, ColorPrint, norm_H1, norm_L2, data_over_line
from meshes import generate_mesh_with_crack


comm = MPI.COMM_WORLD


with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Get parameters from terminal
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_name", default="", type=str, dest="output_name", help="output_name"
)
parser.add_argument("--ud_min", default=0.0, type=float, dest="ud_min", help="ud_min")
parser.add_argument("--ud_max", default=0.02, type=float, dest="ud_max", help="ud_max")
parser.add_argument("--n_steps", default=50, type=int, dest="n_steps", help="n_steps")
parser.add_argument("--k_res", default=2.0e-5, type=float, dest="k_res", help="k_res")
parser.add_argument("--H", default=0.185, type=float, dest="H", help="H")
parser.add_argument("--h_div", default=5.0, type=float, dest="h_div", help="h_div")
parser.add_argument("--ell", default=0.025, type=float, dest="ell", help="ell")
parser.add_argument("--w1", default=0.002, type=float, dest="w1", help="w1")
parser.add_argument("--nu", default=0.99, type=float, dest="nu", help="nu")
parser.add_argument("--E", default=399, type=float, dest="E", help="E")
parser.add_argument("--tdim", default=2, type=int, dest="tdim", help="tdim")
parser.add_argument(
    "--model_dimension",
    default=2,
    type=int,
    dest="model_dimension",
    help="model_dimension",
)
parser.add_argument(
    "--sigma_p",
    default=1,
    type=float,
    dest="sigma_p",
    help="critical plastic stress",
)
parser.add_argument(
    "--c1",
    default=10,
    type=float,
    dest="c1",
    help="plastic-softening",
)
parser.add_argument(
    "--pc_type",
    default="lu",
    type=str,
    dest="pc_type",
    help="elastic_linear_solver",
)

args = parser.parse_args()

# Get mesh parameters
L = parameters["geometry"]["L"]
H = args.H
parameters["geometry"]["H"] = H
tdim = args.tdim
parameters["geometry"]["geometric_dimension"] = tdim
model_dimension = args.model_dimension
parameters["model"]["model_dimension"] = model_dimension
ell_ = args.ell
parameters["model"]["ell"] = ell_
E = args.E
parameters["model"]["E"] = E
nu = args.nu
parameters["model"]["nu"] = nu
w1 = args.w1
parameters["model"]["w1"] = w1
c1 = args.c1
parameters["model"]["c1"] = c1
h_div = args.h_div
parameters["geometry"]["h_div"] = h_div
lc = ell_ / h_div

output_name = args.output_name

# Get geometry model parameters
ud_min = args.ud_min
parameters["loading"]["ud_min"] = ud_min
ud_max = args.ud_max
parameters["loading"]["ud_max"] = ud_max
n_steps = args.n_steps
parameters["loading"]["n_steps"] = n_steps
k_res = args.k_res
parameters["model"]["k_res"] = k_res
lmbda = E * nu / ((1 + nu) * (1 - (model_dimension - 1) * nu))
parameters["model"]["lmbda"] = lmbda
mu = E / (2 * (1 + nu))
parameters["model"]["mu"] = mu
sigma_p = args.sigma_p
parameters["model"]["sigma_p"] = sigma_p
k = lmbda + 2 * mu / model_dimension
parameters["model"]["kappa"] = k

e_par = (lmbda / (3 * mu)) ** (1 / 2) * (H) / (L)
parameters["model"]["e_par"] = e_par

if args.pc_type == "gamg":
    parameters["solvers"]["elasticity"]["snes"]["pc_type"] = args.pc_type
    parameters["solvers"]["elasticity"]["snes"]["ksp_type"] = "cg"
    parameters["solvers"]["elasticity"]["snes"]["ksp_rtol"] = 1.0e-12
    parameters["solvers"]["elasticity"]["snes"]["mg_levels_ksp_type"] = "chebyshev"
    parameters["solvers"]["elasticity"]["snes"]["mg_levels_pc_type"] = "jacobi"
    parameters["solvers"]["elasticity"]["snes"]["mg_levels_esteig_ksp_type"] = "cg"
    parameters["solvers"]["elasticity"]["snes"][
        "mg_levels_ksp_chebyshev_esteig_steps"
    ] = 20
if args.pc_type == "lu":
    parameters["solvers"]["elasticity"]["snes"]["pc_type"] = args.pc_type
    parameters["solvers"]["elasticity"]["snes"]["ksp_type"] = "preonly"
    parameters["solvers"]["elasticity"]["snes"]["pc_factor_mat_solver_type"] = "mumps"

# Create the mesh of the specimen with given dimensions
#gmsh_model, tdim, tag_names = mesh_bar(L, H, lc, tdim)
Lx = 1.
Ly = .5
Lcrack = 0.3
lc =.2
dist_min = .1
dist_max = .3
mesh, mts, fts = generate_mesh_with_crack(
        Lcrack=Lcrack,
        Ly=Ly,
        lc=0.1,  # caracteristic length of the mesh
        refinement_ratio=10,  # how much it is refined at the tip zone
        dist_min=dist_min,  # radius of tip zone
        dist_max=dist_max,  # radius of the transition zone
    )
facet_tags_names = {"left": 1, "right": 2, "top": 3, "crack": 4, "bottom_no_crack": 5}

# Get mesh and meshtags
#mesh, mts = gmsh_model_to_mesh(gmsh_model, cell_data=False, facet_data=True, gdim=tdim)
#interfaces_keys = tag_names["facets"]

outdir = f"output_bar_2D_selective_sigmap0_{sigma_p}_c1_{c1}_w1_{w1}_ell_{ell_}_{output_name}"
prefix = os.path.join(outdir, "output_bar")

if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/parameters.yml", "w") as f:
        yaml.dump(parameters, f)

with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

degree_u = 2
degree_q = 1


def interpolate_quadrature(expr, funct, degree):
    quadrature_points, wts = basix.make_quadrature(basix.CellType.triangle, degree)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = dolfinx.fem.Expression(expr, quadrature_points)
    expr_eval = expr_expr.eval(cells)

    with funct.vector.localForm() as funct_local:
        funct_local.setBlockSize(funct.function_space.dofmap.bs)
        funct_local.setValuesBlocked(
            funct.function_space.dofmap.list.array,
            expr_eval,
            addv=PETSc.InsertMode.INSERT,
        )


def stop_criterion(solver_parameters):
    """Stop criterion for alternate minimization"""
    criterion = solver_parameters.get("criterion")
    criteria = ["residual_u", "alpha_H1", "alpha_L2", "alpha_max", "energy"]

    if criterion in criteria:

        if criterion == "residual_u":
            criterion_residual_u = error_residual_u <= solver_parameters.get(
                "residual_u_tol"
            )
            return criterion_residual_u

        if criterion == "alpha_H1":
            criterion_alpha_H1 = error_alpha_H1 <= solver_parameters.get("alpha_tol")
            return criterion_alpha_H1

        if criterion == "alpha_L2":
            criterion_alpha_L2 = error_alpha_L2 <= solver_parameters.get("alpha_tol")
            return criterion_alpha_L2

        if criterion == "alpha_max":
            criterion_alpha_max = error_alpha_max <= solver_parameters.get("alpha_tol")
            return criterion_alpha_max

        if criterion == "energy":
            criterion_error_energy_r = error_energy_r <= solver_parameters.get(
                "energy_rtol"
            )
            criterion_error_energy_a = error_energy_a <= solver_parameters.get(
                "energy_atol"
            )
            return criterion_error_energy_r or criterion_error_energy_a
    else:
        raise RuntimeError(f"{criterion} is not in {criteria}")


# Measures
dx_quad = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": degree_q, "quadrature_scheme": "default"},
)
dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": 2, "quadrature_scheme": "default"},
)
ds = ufl.Measure(
    "ds",
    subdomain_data=mts,
    domain=mesh,
)

element_u = ufl.FiniteElement("CG", mesh.ufl_cell(), degree=degree_u)
V_u = dolfinx.fem.FunctionSpace(mesh, element_u)

element_alpha = ufl.FiniteElement("CG", mesh.ufl_cell(), degree=1)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

Q_element = ufl.FiniteElement(
    "Quadrature", mesh.ufl_cell(), degree_q, quad_scheme="default"
)
Q = dolfinx.fem.FunctionSpace(mesh, Q_element)

DG_element = ufl.FiniteElement("DG", mesh.ufl_cell(), degree=0)
DG = dolfinx.fem.FunctionSpace(mesh, DG_element)

alpha = dolfinx.fem.Function(V_alpha, name="Damage")
alpha_old = dolfinx.fem.Function(V_alpha)
zero_alpha = dolfinx.fem.Function(V_alpha)

e_p = dolfinx.fem.Function(Q, name="total_plastic_strain")
e_p_bar = dolfinx.fem.Function(Q)
e_p_old = dolfinx.fem.Function(Q)
e_p_bar_old = dolfinx.fem.Function(Q)

e_p_dg = dolfinx.fem.Function(DG, name="total_plastic_strain_dg")
sigma_yy_dg = dolfinx.fem.Function(DG, name="sigma_yy")
trace_sigma_dg = dolfinx.fem.Function(DG, name="trace_sigma")
dev_sigma_dg = dolfinx.fem.Function(DG, name="dev_sigma")
# nucleation_angle_dg = dolfinx.fem.Function(DG, name="nucleation_angle")

u = dolfinx.fem.Function(V_u, name="total_displacement")
u_top = dolfinx.fem.Function(V_u, name="boundary_displacement_top")
u_bottom = dolfinx.fem.Function(V_u, name="boundary_displacement_bottom")

v = ufl.TrialFunction(V_u)
u_ = ufl.TestFunction(V_u)

# need upper/lower bound for the damage field
alpha_lb = dolfinx.fem.Function(V_alpha, name="Lower_bound")
alpha_ub = dolfinx.fem.Function(V_alpha, name="Upper_bound")

alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_lb.x.scatter_forward()
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))
alpha_ub.x.scatter_forward()

num_dofs_global_u = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
num_dofs_global_e_p = Q.dofmap.index_map.size_global * Q.dofmap.index_map_bs
num_dofs_global_alpha = (
    V_alpha.dofmap.index_map.size_global * V_alpha.dofmap.index_map_bs
)
num_dofs_global = num_dofs_global_u + num_dofs_global_alpha + num_dofs_global_e_p

if comm.rank == 0:
    print(f"Number of dofs global: {num_dofs_global}")

bottom_facets = mts.indices[mts.values == facet_tags_names["bottom_no_crack"]]
top_facets = mts.indices[mts.values == facet_tags_names["top"]]

dofs_u_top = dolfinx.fem.locate_dofs_topological(V_u, tdim - 1, np.array(top_facets))
dofs_u_bottom = dolfinx.fem.locate_dofs_topological(
    V_u, tdim - 1, np.array(bottom_facets)
)

dofs_alpha_top = dolfinx.fem.locate_dofs_topological(
    V_alpha, tdim - 1, np.array(top_facets)
)
dofs_alpha_bottom = dolfinx.fem.locate_dofs_topological(
    V_alpha, tdim - 1, np.array(bottom_facets)
)

bcs_u = [
    dolfinx.fem.dirichletbc(u_bottom, dofs_u_bottom),
    dolfinx.fem.dirichletbc(u_top, dofs_u_top),
]

bcs_alpha = [
    dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_bottom),
    dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_top),
]

dolfinx.fem.petsc.set_bc(alpha_ub.vector, bcs_alpha)
alpha_ub.x.scatter_forward()

# Analytical formula for the elastic limit from asymptotic analysis
q_lim = (
    L * sigma_p / (2 * model_dimension * lmbda) * (1 / (1 - 1 / (np.cosh(1 / e_par))))
)

t_lim = H * ufl.sqrt(w1) / (6 * mu * e_par * np.tanh(1 / e_par))


load_par = parameters["loading"]
loads = np.linspace(load_par["ud_min"], load_par["ud_max"], load_par["n_steps"])
Identity = ufl.Identity(model_dimension)


def eps(v):
    return (ufl.grad(v))


def kappa(alpha):
    return k * ((1 - alpha) ** 2 + k_res)


def muf(alpha):
    return mu * ((1 - alpha) ** 2 + k_res)


def pp(alpha):
    return (1 - alpha) ** 2 + k_res


def elastic_energy_density(eps, e_p, alpha):
    return (
        0.5 * kappa(alpha) * (ufl.tr(eps) - e_p) ** 2 * dx_quad
        + muf(alpha) * ufl.inner(ufl.dev(eps), ufl.dev(eps)) * dx
    )


def damage_dissipation_density(alpha):
    grad_alpha = ufl.grad(alpha)
    return (w1 * alpha + w1 * ell_**2 * ufl.dot(grad_alpha, grad_alpha)) * dx


def plastic_dissipation_density(e_p_bar, alpha):
    return sigma_p * pp(alpha) * e_p_bar * dx_quad


def sigma_tr(eps_el, alpha):
    return kappa(alpha) * ufl.tr(eps_el) * Identity


def sigma_dev(eps_el, alpha):
    return 2 * muf(alpha) * ufl.dev(eps_el)


# write now supposing trace_sigma_predictor > 0
def de_p(eps, alpha):
    trace_sigma_predictor = model_dimension * kappa(alpha) * (ufl.tr(eps) - e_p)
    criterion = trace_sigma_predictor / model_dimension - sigma_p * pp(alpha)
    cond1 = ufl.ge(criterion, 0)
    de_p = ufl.conditional(cond1, criterion / (kappa(alpha)), 0)
    return de_p


stress_tr = sigma_tr(
    eps(u) - (e_p + de_p(eps(u), alpha)) * Identity / model_dimension, alpha
)
stress_dev = sigma_dev(eps(u), alpha)
residual_u = ufl.inner(stress_tr, eps(v)) * dx_quad + ufl.inner(stress_dev, eps(v)) * dx

J_u = ufl.derivative(residual_u, u, u_)
solver_u = SNESSolver(
    residual_u,
    u,
    bcs_u,
    J_form=J_u,
    petsc_options=parameters["solvers"]["elasticity"]["snes"],
)

elastic_energy = elastic_energy_density(eps(u), e_p, alpha)
plastic_energy = plastic_dissipation_density(e_p_bar, alpha)
damage_energy = damage_dissipation_density(alpha)
dissipated_energy = plastic_energy + damage_energy
total_energy = elastic_energy + dissipated_energy
energy_alpha = ufl.derivative(total_energy, alpha, ufl.TestFunction(V_alpha))


if parameters.get("solvers").get("damage").get("type") == "SNES":
    solver_alpha = SNESSolver(
        energy_alpha,
        alpha,
        bcs_alpha,
        bounds=(alpha_lb, alpha_ub),
        petsc_options=parameters.get("solvers").get("damage").get("snes"),
        prefix=parameters.get("solvers").get("damage").get("prefix"),
    )
if parameters.get("solvers").get("damage").get("type") == "TAO":
    solver_alpha = TAOSolver(
        total_energy,
        alpha,
        bcs_alpha,
        bounds=(alpha_lb, alpha_ub),
        petsc_options=parameters.get("solvers").get("damage").get("tao"),
        prefix=parameters.get("solvers").get("damage").get("prefix"),
    )
forces = np.zeros_like(loads)

history_data = {
    "load": [],
    "elastic_energy": [],
    "damage_energy": [],
    "plastic_energy": [],
    "dissipated_energy": [],
    "total_energy": [],
    "F": [],
}

for (i, t) in enumerate(loads):

    u_top.interpolate(lambda x: (np.zeros_like(x[0]), (H / L) * t * np.ones_like(x[1])))
    u_bottom.interpolate(
        lambda x: (np.zeros_like(x[0]), -(H / L) * t * np.ones_like(x[1]))
    )
    u_top.x.scatter_forward()
    u_bottom.x.scatter_forward()

    alpha.vector.copy(alpha_old.vector)
    alpha_old.x.scatter_forward()

    # update the lower bound
    alpha.vector.copy(alpha_lb.vector)
    alpha_lb.x.scatter_forward()

    alpha_diff = dolfinx.fem.Function(alpha.function_space)
    residual = dolfinx.fem.Function(u.function_space)
    total_energy_int_old = 0

    ColorPrint.print_bold(f"-- Solving for it = {i}, t = {t:3.5f} --")

    for iteration in range(
        parameters.get("solvers").get("damage_elasticity").get("max_it")
    ):
        e_p_old.vector.copy(e_p.vector)
        e_p.x.scatter_forward()
        e_p_bar_old.vector.copy(e_p_bar.vector)
        e_p_bar.x.scatter_forward()
        # solve non-linear elastoplastic problem
        solver_u.solve()

        e_p.vector.copy(e_p_old.vector)
        e_p_old.x.scatter_forward()
        e_p_bar.vector.copy(e_p_bar_old.vector)
        e_p_bar_old.x.scatter_forward()

        # update the total plastic strain

        interpolate_quadrature(
            e_p_bar + ufl.sqrt(de_p(eps(u), alpha) ** 2), e_p_bar, degree_q
        )
        e_p_bar.x.scatter_forward()

        interpolate_quadrature(e_p + de_p(eps(u), alpha), e_p, degree_q)
        e_p.x.scatter_forward()

        (solver_alpha_it, solver_alpha_reason) = solver_alpha.solve()

        alpha.vector.copy(alpha_diff.vector)
        alpha_diff.vector.axpy(-1, alpha_old.vector)
        alpha_diff.x.scatter_forward()

        error_alpha_H1 = norm_H1(alpha_diff)
        error_alpha_L2 = norm_L2(alpha_diff)
        error_alpha_max = alpha_diff.vector.max()[1]

        solver_u.solver.computeFunction(u.vector, residual.vector)
        u.vector, residual.x.scatter_forward()
        error_residual_u = residual.vector.norm()
        total_energy_int = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(total_energy)),
            op=MPI.SUM,
        )
        elastic_energy_int = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(elastic_energy)),
            op=MPI.SUM,
        )
        damage_energy_int = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(damage_energy)),
            op=MPI.SUM,
        )
        plastic_energy_int = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(plastic_energy)),
            op=MPI.SUM,
        )
        dissipated_energy_int = comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(dissipated_energy)),
            op=MPI.SUM,
        )
        error_energy_a = abs(total_energy_int - total_energy_int_old)

        if total_energy_int_old > 0:
            error_energy_r = abs(total_energy_int / total_energy_int_old - 1)
        else:
            error_energy_r = 1.0

        total_energy_int_old = total_energy_int
        alpha.vector.copy(alpha_old.vector)
        alpha_old.x.scatter_forward()
        ColorPrint.print_info(
            f"AM - Iteration: {iteration:3d}, "
            + f"alpha_max: {alpha.vector.max()[1]:3.4e}, "
            + f"e_p_max: {e_p.vector.max()[1]:3.4e},"
            + f"Error_alpha_max: {error_alpha_max:3.4e}, "
            + f"Error_energy: {error_energy_r:3.4e}, "
            + f"Error_residual_u: {error_residual_u:3.4e}"
        )
        if stop_criterion(parameters["solvers"]["damage_elasticity"]):
            e_p.vector.copy(e_p_old.vector)
            e_p_old.x.scatter_forward()
            e_p_bar.vector.copy(e_p_bar_old.vector)
            e_p_bar_old.x.scatter_forward()
            break
    else:
        if (
            not parameters["solvers"]
            .get("damage_elasticity")
            .get("error_on_nonconvergence")
        ):
            ColorPrint.print_warn(
                (
                    f"Could not converge after {iteration:3d} iterations,"
                    + f"error_u {error_residual_u:3.4e},"
                    + f"error_alpha_max {error_alpha_max:3.4e},"
                    + f"error_energy_r {error_energy_r:3.4e}"
                )
            )
            e_p.vector.copy(e_p_old.vector)
            e_p_old.x.scatter_forward()
            e_p_bar.vector.copy(e_p_bar_old.vector)
            e_p_bar_old.x.scatter_forward()

        else:
            raise RuntimeError(
                f"Could not converge after {iteration:3d} iterations, error {error_residual_u:3.4e}"
            )

    project(e_p, e_p_dg, dx_quad)
    e_p_dg.x.scatter_forward()

    stress = sigma_tr(
        eps(u) - (e_p_dg) * Identity / model_dimension, alpha
    ) + sigma_dev(eps(u), alpha)

    trace_sigma_expr = dolfinx.fem.Expression(
        ufl.tr(stress), DG.element.interpolation_points
    )
    sigma_yy_expr = dolfinx.fem.Expression(
        stress[1, 1], DG.element.interpolation_points
    )
    dev_sigma_expr = dolfinx.fem.Expression(
        ufl.sqrt(ufl.inner(ufl.dev(stress), ufl.dev(stress))),
        DG.element.interpolation_points,
    )
    """nucleation_angle_expr = dolfinx.fem.Expression(
        ufl.atan_2(dev_sigma_expr, trace_sigma_expr) * 180 / np.pi,
        DG.element.interpolation_points,
    )"""

    sigma_yy_dg.interpolate(sigma_yy_expr)
    sigma_yy_dg.x.scatter_forward()
    forces[i] = comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(sigma_yy_dg * ds(1))
        ),
        op=MPI.SUM,
    )
    trace_sigma_dg.interpolate(trace_sigma_expr)
    trace_sigma_dg.x.scatter_forward()
    dev_sigma_dg.interpolate(dev_sigma_expr)
    dev_sigma_dg.x.scatter_forward()
    """nucleation_angle_dg.interpolate(nucleation_angle_expr)
    nucleation_angle_dg.x.scatter_forward()"""
    #
    # ----------------
    # Post-processing
    # ----------------
    import matplotlib.pyplot as plt

    t_e = loads[3]
    if t == t_e:
        print(L, H, e_par)
        tol = 0.0001  # Avoid hitting the outside of the domain
        x = np.linspace(0 + tol, L - tol, 31)
        x1 = np.linspace(0 + tol, L - tol, 3)
        y1 = np.linspace(0.0, 0.0, 1)
        points = np.zeros((3, 31))
        points[0] = x
        points[1] = H / 2 - tol
        fig_2 = plt.figure()
        points_on_y, p_val = data_over_line(points, trace_sigma_dg)
        if comm.rank == 0:
            points_on_y0 = points_on_y.T[:, 0]
            p_val0 = p_val[0:31]
            plt.plot(
                points_on_y0,
                p_val0,
                color="#E63946",
                marker="o",
                linewidth=2,
                label="Finite Element",
            )
        plt.plot(
            x,
            lmbda
            * 4
            * t
            * (
                1
                - np.cosh(np.sqrt(3 * mu / lmbda) * 2 * (x - L / 2) / H)
                / np.cosh(1 / e_par)
            )
            / L,
            color="#1D2769",
            linewidth=2,
            label="Asymptotic",
        )

        # plt.title("sig_12 on y=H")
        plt.grid(True)
        plt.legend(loc=(0.2, -0.47), framealpha=1, fontsize=20)
        plt.axhline(
            y=lmbda * 4 * t * (1 - 1 / np.cosh(1 / e_par)) / L,
            color="#457B9D",
            linestyle="--",
            linewidth=2,
            label=r"$2\lambda_0 \frac{\Delta}{L}\left(1-\frac{1}{cosh(\frac{1}{e})}\right)$",
        )
        plt.text(
            -1.1,
            8.5,
            r"$2\lambda_0 \frac{\Delta}{L}\left(1-\frac{1}{cosh\left(\frac{1}{e}\right)}\right)$",
            fontsize=20,
            color="#457B9D",
        )
        plt.xticks(ticks=x1, labels=["-L", "0", "L"], fontsize=20)
        plt.yticks(ticks=y1, labels=["0"], fontsize=20)
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"tr$(\sigma)$ on $x_2 = 0$", fontsize=20)
        plt.savefig(f"{outdir}/p_middle_horizontal.png", bbox_inches="tight")
    plt.figure()
    plt.plot(loads, 2 * loads * lmbda * (1 - e_par * np.tanh(1 / e_par)))
    plt.plot(loads, 4 * loads * lmbda * (1 - 2 * e_par * np.tanh(1 / (2 * e_par))))
    plt.plot(loads, forces, "o-c")
    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.savefig(f"{prefix}-force.png")
    plt.close("all")

    history_data["load"].append(t)
    history_data["dissipated_energy"].append(dissipated_energy_int)
    history_data["elastic_energy"].append(elastic_energy_int)
    history_data["plastic_energy"].append(plastic_energy_int)
    history_data["damage_energy"].append(damage_energy_int)
    history_data["total_energy"].append(total_energy_int)
    history_data["F"].append(forces[i])

    if comm.rank == 0:
        a_file = open(f"{prefix}_data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

    with XDMFFile(comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)
        file.write_function(e_p_dg, t)
        file.write_function(trace_sigma_dg, t)
        file.write_function(dev_sigma_dg, t)
