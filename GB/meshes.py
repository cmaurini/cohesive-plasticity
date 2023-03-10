import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
import dolfinx.plot


def generate_mesh_with_crack(
    Lx=1.0,
    Ly=0.5,
    Lcrack=0.3,
    lc=0.1,
    dist_min=0.1,
    dist_max=0.3,
    refinement_ratio=10,
    gdim=2,
    verbosity=10
):

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gmsh.initialize()

    facet_tags = {"left": 1, "right": 2, "top": 3, "crack": 4, "bottom_no_crack": 5}
    cell_tags = {"all": 20}

    if mesh_comm.rank == model_rank:
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        # Create the points
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc)
        p2 = model.geo.addPoint(Lcrack, 0.0, 0, lc)
        p3 = model.geo.addPoint(Lx, 0, 0, lc)
        p4 = model.geo.addPoint(Lx, Ly, 0, lc)
        p5 = model.geo.addPoint(0, Ly, 0, lc)
        # Create the lines
        l1 = model.geo.addLine(p1, p2, tag=facet_tags["crack"])
        l2 = model.geo.addLine(p2, p3, tag=facet_tags["bottom_no_crack"])
        l3 = model.geo.addLine(p3, p4, tag=facet_tags["right"])
        l4 = model.geo.addLine(p4, p5, tag=facet_tags["top"])
        l5 = model.geo.addLine(p5, p1, tag=facet_tags["left"])
        # Create the surface
        cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4, l5])
        surface_1 = model.geo.addPlaneSurface([cloop1])

        # Define the mesh size and fields for the mesh refinement
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [p2])
        # SizeMax -                   / ------------------
        #                            /
        # SizeMin -o----------------/
        #          |                |  |
        #        Point        DistMin   DistMax
        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", lc / refinement_ratio)
        model.mesh.field.setNumber(2, "LcMax", lc)
        model.mesh.field.setNumber(2, "DistMin", dist_min)
        model.mesh.field.setNumber(2, "DistMax", dist_max)
        model.mesh.field.setAsBackgroundMesh(2)
        model.geo.synchronize()

        # Assign mesh and facet tags
        surface_entities = [entity[1] for entity in model.getEntities(2)]
        model.addPhysicalGroup(2, surface_entities, tag=cell_tags["all"])
        model.setPhysicalName(2, 2, "Rectangle surface")
        gmsh.option.setNumber('General.Verbosity', verbosity)
        model.mesh.generate(gdim)

        for (key, value) in facet_tags.items():
            model.addPhysicalGroup(1, [value], tag=value)
            model.setPhysicalName(1, value, key)

        msh, cell_tags, facet_tags = gmshio.model_to_mesh(
            model, mesh_comm, model_rank, gdim=gdim
        )
        gmsh.finalize()
        msh.name = "rectangle"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"
        return msh, cell_tags, facet_tags

    
def generate_cylinder(
    h=5,
    R=1,
    lc=0.1,
    dist_min=0.1,
    dist_max=0.3,
    refinement_ratio=1,
    gdim=3,
    verbosity=10
):

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gmsh.initialize()

    facet_tags = {"bottom": 1, "top": 2, "lateral": 3}
    cell_tags = {"all": 11}
    
    if mesh_comm.rank == model_rank:

        model = gmsh.model()

        cylinder = model.occ.addCylinder(0, 0, 0, 0, 0, h, R) # 3 coor of center, 3 coor of axis, 1 radius
        gmsh.model.occ.synchronize()

        # On récupère le volume
        # ---------------------------------------------------------------------------
        volume = gmsh.model.getEntities(dim=3)
        #assert(volume == cylinder[0])
        gmsh.model.addPhysicalGroup(volume[0][0], [volume[0][1]], cell_tags["all"])
        gmsh.model.setPhysicalName(volume[0][0], cell_tags["all"], "Cylinder volume")

        # On récupère les surfaces
        # ---------------------------------------------------------------------------
        surfaces = gmsh.model.occ.getEntities(dim=2)
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [0, 0, 0]): # bottom surface
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], facet_tags["bottom"])
                bottom = surface[1]
                gmsh.model.setPhysicalName(surface[0], facet_tags["bottom"], "Cylinder bottom")
            elif np.allclose(com, [0, 0, h]): # top surface
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], facet_tags["top"])
                top = surface[1]
                gmsh.model.setPhysicalName(surface[0], facet_tags["top"], "Cylinder top")
            else:
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], facet_tags["lateral"])
                lateral = surface[1]
                gmsh.model.setPhysicalName(surface[0], facet_tags["lateral"], "Cylinder lateral")

        # Mesh resolution
        # ---------------------------------------------------------------------------
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "FacesList", [top, bottom, lateral]) 
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "LcMin", lc/refinement_ratio)
        gmsh.model.mesh.field.setNumber(threshold, "LcMax", lc)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('General.Verbosity', verbosity)

        # Mesh generation
        # ---------------------------------------------------------------------------
        gmsh.model.mesh.generate(gdim)


        msh, cell_tags, facet_tags = gmshio.model_to_mesh(model, mesh_comm, model_rank, gdim=gdim)
        gmsh.finalize()
        msh.name = "cylinder"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"
        return msh, cell_tags, facet_tags

    
def generate_hollow_cylinder_2D(
    R_i=0.5,
    R_e=1.,
    lc=0.1,
    dist_min=0.1,
    dist_max=0.3,
    refinement_ratio=1,
    gdim=2,
    verbosity=10
):
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gmsh.initialize()
    facet_tags = {"inner_boundary": 1, "outer_boundary": 2}
    cell_tags = {"all": 11}
    if mesh_comm.rank == model_rank:
        
        # Stating the geometry
        # ---------------------------------------------------------------------------
        model = gmsh.model()
        model.add("Disk")
        model.setCurrent("Disk")
        gdim = gdim # geometric dimension of the mesh (shall be 2 since we are in anti-plane elasticity)
        inner_disk = gmsh.model.occ.addDisk(0, 0, 0, R_i, R_i)
        outer_disk = gmsh.model.occ.addDisk(0, 0, 0, R_e, R_e)
        whole_domain = gmsh.model.occ.cut([(gdim, outer_disk)], [(gdim, inner_disk)])
        gmsh.model.occ.synchronize()

        # Add physical tag for bulk
        # ---------------------------------------------------------------------------
        volume = gmsh.model.getEntities(dim=gdim)
        gmsh.model.addPhysicalGroup(volume[0][0], [volume[0][1]], cell_tags["all"])
        gmsh.model.setPhysicalName(volume[0][0], cell_tags["all"], "Cylinder cross surface")

        # Add physical tag for boundaries
        # ---------------------------------------------------------------------------
        lines = gmsh.model.getEntities(dim=1)
        inner_boundary = lines[1][1]
        outer_boundary = lines[0][1]
        gmsh.model.addPhysicalGroup(1, [inner_boundary], facet_tags["inner_boundary"])
        gmsh.model.addPhysicalGroup(1, [outer_boundary], facet_tags["outer_boundary"])
        
        # Mesh resolution
        # ---------------------------------------------------------------------------
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "FacesList", [inner_boundary]) # when refining the mesh, we want it thinner at the inner boundary
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "LcMin", lc/refinement_ratio)
        gmsh.model.mesh.field.setNumber(threshold, "LcMax", lc)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('General.Verbosity', verbosity)
        
        # Mesh generation
        # ---------------------------------------------------------------------------
        model.mesh.generate(gdim)
        gmsh.option.setNumber("General.Terminal", 1)
        model.mesh.setOrder(2) # mesh order
        gmsh.option.setNumber("General.Terminal", 0)

        # Import the mesh in dolfinx
        # ---------------------------------------------------------------------------
        msh, cell_tags, facet_tags = gmshio.model_to_mesh(model, mesh_comm, model_rank, gdim=gdim)
        msh.name = "hollow_cylinder_2D"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"
        gmsh.finalize()
        return msh, cell_tags, facet_tags

    


def generate_PacMan(
    R=1.,
    theta=np.pi/4,
    lc=0.1,
    dist_min=0.1,
    dist_max=0.3,
    refinement_ratio=1,
    gdim=2,
    verbosity=10
):
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gmsh.initialize()
    facet_tags = {"Top_lip": 3, "Bottom_lip": 2, "Disk": 1}
    cell_tags = {"all": 20}
    if mesh_comm.rank == model_rank:
        model = gmsh.model()
        model.add("PacMan")
        model.setCurrent("PacMan")
        # Create the points
        h = R*np.tan(theta/2)
        p1 = model.occ.addPoint(0.0, 0.0, 0, lc, 1)
        p2 = model.occ.addPoint(-R, -h, 0, lc)
        p3 = model.occ.addPoint(-R, h, 0, lc)
        # Create the lines
        l1 = model.occ.addLine(p1, p2)
        l2 = model.occ.addLine(p2, p3)
        l3 = model.occ.addLine(p3, p1)
        # Create the surface
        cloop1 = model.occ.addCurveLoop([l1, l2, l3])
        surface_1 = model.occ.addPlaneSurface([cloop1])

        disk = model.occ.addDisk(0, 0, 0, R, R)
        whole_domain = model.occ.cut([(gdim, disk)], [(gdim, surface_1)])

        # Define the mesh size and fields for the mesh refinement
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "PointsList", [p3]) # p3 fonctionne mais en soit c'est juste qu'on ne connaît pas l'indice du centre du disque...
                # SizeMax -                   / ------------------
                #                            /
                # SizeMin -o----------------/
                #          |                |  |
                #        Point        DistMin   DistMax
        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", lc / refinement_ratio)
        model.mesh.field.setNumber(2, "LcMax", lc)
        model.mesh.field.setNumber(2, "DistMin", dist_min)
        model.mesh.field.setNumber(2, "DistMax", dist_max)
        model.mesh.field.setAsBackgroundMesh(2)
        model.occ.synchronize()
        # Assign mesh and facet tags
        surface_entities = [entity[1] for entity in model.getEntities(2)]
        model.addPhysicalGroup(2, surface_entities, tag=cell_tags["all"])    
        model.setPhysicalName(2, 2, "PacMan surface")

        gmsh.option.setNumber('General.Verbosity', verbosity)
        model.mesh.generate(gdim)

        for (key, value) in facet_tags.items():
                model.addPhysicalGroup(1, [value], tag=value)
                model.setPhysicalName(1, value, key)

        msh, cell_tags, facet_tags = gmshio.model_to_mesh(
        model, mesh_comm, model_rank, gdim=gdim
        )
        gmsh.finalize()
        msh.name = "PacMan"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"
        return msh, cell_tags, facet_tags