from mpi4py import MPI
import ufl
import dolfinx
from petsc4py import PETSc
import sys
import petsc4py

petsc4py.init(sys.argv)

from dolfinx.cpp.log import LogLevel, log
from dolfinx.fem import form

from utils import ColorPrint

comm = MPI.COMM_WORLD


class SNESSolver:
    """
    Problem class for elasticity, compatible with PETSC.SNES solvers.
    """

    def __init__(
        self,
        F_form: ufl.Form,
        u: dolfinx.fem.Function,
        bcs=[],
        J_form: ufl.Form = None,
        bounds=None,
        petsc_options={},
        form_compiler_parameters={},
        jit_parameters={},
        monitor=None,
        prefix=None,
    ):

        self.u = u
        self.bcs = bcs
        self.bounds = bounds

        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "snes_{}".format(str(id(self))[0:4])

        self.prefix = prefix

        if self.bounds is not None:
            self.lb = bounds[0]
            self.ub = bounds[1]

        V = self.u.function_space
        self.comm = V.mesh.comm
        self.F_form = dolfinx.fem.form(F_form)

        if J_form is None:
            J_form = ufl.derivative(F_form, self.u, ufl.TrialFunction(V))

        self.J_form = dolfinx.fem.form(J_form)

        self.petsc_options = petsc_options

        self.monitor = monitor
        self.solver = self.solver_setup()

    def set_petsc_options(self, debug=False):
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        if debug is True:
            ColorPrint.print_info(self.petsc_options)

        for k, v in self.petsc_options.items():
            opts[k] = v

        opts.prefixPop()

    def solver_setup(self):
        # Create nonlinear solver
        snes = PETSc.SNES().create(self.comm)

        # Set options
        snes.setOptionsPrefix(self.prefix)
        self.set_petsc_options()
        snes.setFromOptions()

        self.b = dolfinx.fem.petsc.create_vector(self.F_form)
        self.a = dolfinx.fem.petsc.create_matrix(self.J_form)

        snes.setFunction(self.F, self.b)
        snes.setJacobian(self.J, self.a)

        # We set the bound (Note: they are passed as reference and not as values)

        if self.monitor is not None:
            snes.setMonitor(self.monitor)

        if self.bounds is not None:
            snes.setVariableBounds(self.lb.vector, self.ub.vector)

        return snes

    #   def default_parameters(self):

    #        with open("default_parameters.yml") as f:
    #            parameters = yaml.load(f, Loader=yaml.FullLoader)

    #            self.default_parameters = parameters.get("solver").get(
    #                f"{self.solver_name}"
    #            )

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # We need to assign the vector to the function
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.x.scatter_forward()

        # Zero the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, self.F_form)

        # Apply boundary conditions
        dolfinx.fem.petsc.apply_lifting(b, [self.J_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.J_form, self.bcs)
        A.assemble()

    def solve(self):
        log(LogLevel.INFO, f"Solving {self.prefix}")
        # self.solver_setup()
        try:
            self.solver.solve(None, self.u.vector)
            # print(
            #    f"{self.prefix} SNES solver converged in",
            #    self.solver.getIterationNumber(),
            #    "iterations",
            #    "with converged reason",
            #    self.solver.getConvergedReason(),
            # )
            self.u.x.scatter_forward()
            return (self.solver.getIterationNumber(), self.solver.getConvergedReason())

        except Warning:
            log(
                LogLevel.WARNING,
                f"WARNING: {self.prefix} solver failed to converge, what's next?",
            )
            raise RuntimeError(f"{self.prefix} solvers did not converge")


class TAOSolver(SNESSolver):
    """
    Problem class for elasticity, compatible with PETSC.SNES solvers.
    """

    def __init__(
        self,
        total_energy,
        u,
        bcs,
        bounds=None,
        petsc_options={},
        form_compiler_parameters={},
        jit_parameters={},
        monitor=None,
        prefix=None,
    ):

        self.functional = total_energy
        self.u = u

        F_form = ufl.derivative(
            self.functional, self.u, ufl.TestFunction(self.u.function_space)
        )
        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "tao_ds{}".format(str(id(self))[0:4])

        super().__init__(
            F_form,
            u,
            bcs,
            bounds=bounds,
            petsc_options=petsc_options,
            monitor=monitor,
            prefix=prefix,
        )

    def solver_setup(self):
        # Create nonlinear solver
        self.b = dolfinx.fem.petsc.create_vector(self.F_form)
        self.a = dolfinx.fem.petsc.create_matrix(self.J_form)
        tao = PETSc.TAO().create(self.comm)
        tao.setOptionsPrefix(self.prefix)
        self.set_petsc_options()

        tao.setObjective(self.value)
        tao.setGradient(self.F, self.b)
        tao.setHessian(self.J, self.a)

        if self.monitor:
            tao.setMonitor(self.monitor)

        if self.bounds is not None:
            tao.setVariableBounds(self.lb.vector, self.ub.vector)

        tao.setFromOptions()

        return tao

    def value(self, tao, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        return comm.allreduce(
            dolfinx.fem.assemble_scalar(form(self.functional)), op=MPI.SUM
        )

    def solve(self):
        log(LogLevel.INFO, f"Solving {self.prefix}")

        try:
            self.solver.solve(self.u.vector)
            return (self.solver.getIterationNumber(), self.solver.getConvergedReason())

        except Warning:
            log(
                LogLevel.WARNING,
                f"WARNING: {self.prefix} solver failed to converge, what's next?",
            )
            raise RuntimeError(f"{self.prefix} solvers did not converge")
