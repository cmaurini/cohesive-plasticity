import sys
from contextlib import ExitStack

import dolfinx.la as _la
import mpi4py
import numpy as np
import ufl
from pathlib import Path
from dolfinx import geometry
from dolfinx.fem import (
    assemble_scalar,
    Function,
    FunctionSpace,
    form,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    set_bc,
)

from mpi4py import MPI
from petsc4py import PETSc


comm = MPI.COMM_WORLD


def project(v, target_func, dx=None, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    if not dx:
        dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = form(ufl.inner(Pv, w) * dx)
    L = form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
    target_func.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )


def print0(str):
    """Prints on root node"""
    import mpi4py.MPI.COMM_WORLD as comm

    if comm.rank == 0:
        print(str)
    return


class ColorPrint:

    """
    Colored printing functions for strings that use universal ANSI escape
    sequences.
        - fail: bold red
        - pass: bold green,
        - warn: bold yellow,
        - info: bold blue
        - color: bold cyan
        - bold: bold white
    """

    @staticmethod
    def print_fail(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)
            sys.stderr.flush()

    @staticmethod
    def print_pass(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_warn(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)
            sys.stderr.flush()

    @staticmethod
    def print_info(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;34m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_color(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;36m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_bold(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;37m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()


def norm_L1(u):
    """
    Returns the L2 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(np.abs(u) * dx)
    norm = comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM)
    return norm


def norm_L2(u):
    """
    Returns the L1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(ufl.inner(u, u) * dx)
    norm = np.sqrt(comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def norm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form((ufl.inner(u, u) + ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    norm = np.sqrt(comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def extract_linear_combination(e, linear_comb=[], scalar_weight=1.0):
    """Extract linear combination from UFL expression.
    Assumes the expression could be equivalently written as ``\\sum_i c_i u_i``
    where ``c_i`` are known scalar coefficients and ``u_i`` are dolfinx Functions.
    If this assumption fails, raises a RuntimeError.
    Returns
    -------
    Tuples (u_i, c_i) which represent summands in the above sum.
    Returned summands are not uniquely accumulated, i.e. could return (u, 1.0) and (u, 2.0).
    Note
    ----
    Constant nodes (dolfinx.Constant) are not handled. So the expression which has the above
    form where ``c_i`` could contain Constant must have first these nodes numerically evaluated.
    """
    from ufl.classes import Division, Product, ScalarValue, Sum

    if isinstance(e, Function):
        linear_comb.append((e, scalar_weight))
    elif isinstance(e, (Product, Division)):
        assert len(e.ufl_operands) == 2

        if isinstance(e.ufl_operands[0], ScalarValue):
            scalar = e.ufl_operands[0]
            expr = e.ufl_operands[1]
        elif isinstance(e.ufl_operands[1], ScalarValue):
            scalar = e.ufl_operands[1]
            expr = e.ufl_operands[0]
        else:
            raise RuntimeError(f"One operand of {type(e)} must be ScalarValue")

        if isinstance(e, Product):
            scalar_weight *= float(scalar)
        else:
            scalar_weight /= float(scalar)

        extract_linear_combination(expr, linear_comb, scalar_weight)
    elif isinstance(e, Sum):
        e0 = e.ufl_operands[0]
        e1 = e.ufl_operands[1]
        extract_linear_combination(e0, linear_comb, scalar_weight)
        extract_linear_combination(e1, linear_comb, scalar_weight)
    else:
        raise RuntimeError(f"Expression type {type(e)} not handled")

    return linear_comb


def data_over_line(points, function):

    mesh = function.function_space.mesh
    comm = mesh.comm
    # comm
    bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)

    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc)

    if len(points_on_proc) > 0:
        values_on_proc = function.eval(points_on_proc, cells)
        point_data_proc = [points_on_proc.T, values_on_proc]
    else:
        point_data_proc = None

    point_data = comm.gather(point_data_proc, root=0)

    if comm.rank == 0:
        point_data = list(filter(None, point_data))
        points = np.concatenate([data_proc[0].T for data_proc in point_data])
        values = np.concatenate([data_proc[1] for data_proc in point_data])
    else:
        points = None
        values = None
    return points, values


import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_mesh(mesh, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax


def plot_mesh_tags(mesh_tags, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    mesh = mesh_tags.mesh
    points = mesh.geometry.x
    colors = ["b", "r"]
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap_bounds = [0, 0.5, 1]
    norm = matplotlib.colors.BoundaryNorm(cmap_bounds, cmap.N)
    assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)
    if mesh_tags.dim == mesh.topology.dim:
        cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        cell_colors = np.zeros((cells.shape[0],))
        cell_colors[mesh_tags.indices[mesh_tags.values == 1]] = 1
        mappable = ax.tripcolor(tria, cell_colors, edgecolor="k", cmap=cmap, norm=norm)
    elif mesh_tags.dim == mesh.topology.dim - 1:
        tdim = mesh.topology.dim
        geometry_dofmap = mesh.geometry.dofmap
        cells_map = mesh.topology.index_map(mesh.topology.dim)
        num_cells = cells_map.size_local + cells_map.num_ghosts
        connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
        connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
        connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
        vertex_map = {
            topology_index: geometry_index
            for c in range(num_cells)
            for (topology_index, geometry_index) in zip(
                connectivity_cells_to_vertices.links(c), geometry_dofmap.links(c)
            )
        }
        linestyles = [(0, (5, 10)), "solid"]
        lines = list()
        lines_colors_as_int = list()
        lines_colors_as_str = list()
        lines_linestyles = list()
        mesh_tags_1 = mesh_tags.indices[mesh_tags.values == 1]
        for c in range(num_cells):
            facets = connectivity_cells_to_facets.links(c)
            for f in facets:
                if f in mesh_tags_1:
                    value_f = 1
                else:
                    value_f = 0
                vertices = [
                    vertex_map[v] for v in connectivity_facets_to_vertices.links(f)
                ]
                lines.append(points[vertices][:, :2])
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(colors[value_f])
                lines_linestyles.append(linestyles[value_f])
        mappable = matplotlib.collections.LineCollection(
            lines,
            cmap=cmap,
            norm=norm,
            colors=lines_colors_as_str,
            linestyles=lines_linestyles,
        )
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(mappable)
        ax.autoscale()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(
        mappable,
        cax=cax,
        cmap=cmap,
        norm=norm,
        boundaries=cmap_bounds,
        ticks=cmap_bounds,
    )
    return ax
    f


def _plot_dofmap(coordinates, ax=None):
    if ax is None:
        ax = plt.gca()
    text_offset = [1e-2, 1e-2]
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c="k", s=50)
    for c in np.unique(coordinates, axis=0):
        dofs_c = (coordinates == c).all(axis=1).nonzero()[0]
        text_c = np.array2string(dofs_c, separator=", ", max_line_width=10)
        ax.text(c[0] + text_offset[0], c[1] + text_offset[1], text_c, fontsize=10)
    return ax


def plot_dofmap(V, ax=None):
    coordinates = V.tabulate_dof_coordinates().round(decimals=3)
    return _plot_dofmap(coordinates, ax)


def set_vector_to_constant(x, value):
    with x.localForm() as local:
        local.set(value)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# # viz helpers

# import dolfinx
# import pyvista


# def plot_vector(u, plotter, subplot=None):
#     if subplot:
#         plotter.subplot(subplot[0], subplot[1])
#     V = u.function_space
#     mesh = V.mesh
#     topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
#     num_dofs_local = u.function_space.dofmap.index_map.size_local
#     geometry = u.function_space.tabulate_dof_coordinates()[:num_dofs_local]
#     values = np.zeros((V.dofmap.index_map.size_local, 3), dtype=np.float64)
#     values[:, : mesh.geometry.dim] = u.vector.array.real.reshape(
#         V.dofmap.index_map.size_local, V.dofmap.index_map_bs
#     )
#     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#     grid["vectors"] = values
#     grid.set_active_vectors("vectors")
#     # geom = pyvista.Arrow()
#     # glyphs = grid.glyph(orient="vectors", factor=1, geom=geom)
#     glyphs = grid.glyph(orient="vectors", factor=1.0)
#     plotter.add_mesh(glyphs)
#     plotter.add_mesh(
#         grid, show_edges=True, color="black", style="wireframe", opacity=0.3
#     )
#     plotter.view_xy()
#     return plotter
#     # figure = plotter.screenshot(f"./output/test_viz/test_viz_MPI{comm.size}-.png")


# def plot_scalar(alpha, plotter, subplot=None):
#     if subplot:
#         plotter.subplot(subplot[0], subplot[1])
#     V = alpha.function_space
#     mesh = V.mesh
#     topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)

#     plotter.subplot(0, 0)
#     grid.point_arrays["alpha"] = alpha.compute_point_values().real
#     grid.set_active_scalars("alpha")
#     plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, clim=[0, 1])
#     plotter.view_xy()
#     return plotter


def build_nullspace_elasticity(V: FunctionSpace):
    """
    Function to build nullspace for 2D/3D elasticity.
    Parameters:
    ===========
    V
        The function space
    """
    _x = Function(V)
    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [_x.vector.copy() for _ in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        dofs = [V.sub(i).dofmap.list.array for i in range(gdim)]

        # Build translational null space basis
        for i in range(gdim):
            basis[i][dofs[i]] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        if gdim == 2:
            basis[2][dofs[0]] = -x1
            basis[2][dofs[1]] = x0
        elif gdim == 3:
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0

            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2

    _la.orthonormalize(nullspace_basis)
    assert _la.is_orthonormal(nullspace_basis)
    return PETSc.NullSpace().create(vectors=nullspace_basis)
