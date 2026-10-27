import gmsh
import numpy as np

def compareLines(Line_A,Line_B,tol=0.2):
    """
    Checks if Line_A curve is near Line_B curve and if so then stores Line_A curve
    """
    gmsh.model.occ.synchronize()
    Fin_List = []
    for ii in Line_A:
        for jj in Line_B:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, ii)
            A_cen = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, jj)
            B_cen = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
            if np.linalg.norm(A_cen-B_cen)<tol:
                Fin_List.append(ii)

    return list(set(Fin_List))


def updateVolList(Imp_list, OG_vol, Fin_vol):
    Fin_list = []
    for sublist in Imp_list:
        if isinstance(sublist,list):
            A_list = []
            for kk in sublist:
                idx = OG_vol.index((3,kk))
                A_list.append(Fin_vol[idx][0][1])
            Fin_list.append(A_list)
        else:
            idx = OG_vol.index((3,sublist))
            Fin_list.append(Fin_vol[idx][0][1])
    return Fin_list 

def updateSurfList(Imp_list, OG_Surf, Fin_Surf):
    Fin_list = []
    for sublist in Imp_list:
        if isinstance(sublist,list):
            A_list = []
            for kk in sublist:
                idx = OG_Surf.index((2,kk))
                A_list.append(Fin_Surf[idx][0][1])
            Fin_list.append(A_list)
        else:
            idx = OG_Surf.index((2,sublist))
            Fin_list.append(Fin_Surf[idx][0][1])
    return Fin_list 

def genCircZ(radius):
    surface = gmsh.model.occ.addDisk(
        0, 0, 0,
        radius, radius
    )

    return surface

def rotateVol(volume, axis, angle):
    axis = axis.lower()

    if axis == "x":
        direction = (1, 0, 0)
    elif axis == "y":
        direction = (0, 1, 0)
    elif axis == "z":
        direction = (0, 0, 1)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    gmsh.model.occ.rotate(
        [(3, volume)],
        0, 0, 0,
        *direction,
        np.radians(angle)
    )



def rotateSurf(surface, axis, angle):
    axis = axis.lower()

    if axis == "x":
        direction = (1, 0, 0)
    elif axis == "y":
        direction = (0, 1, 0)
    elif axis == "z":
        direction = (0, 0, 1)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    gmsh.model.occ.rotate(
        [(2, surface)],
        0, 0, 0,              # point on rotation axis
        *direction,           # rotation axis
        np.radians(angle)   # angle in radians
    )

def scaleVol(volume, x=1, y=1, z=1):
    gmsh.model.occ.dilate(
        [(3, volume)],
        0, 0, 0,   # center of scaling
        x, y, z
    )


def sweepSurf(surface, axis, angle):
    axis = axis.lower()

    if axis == "x":
        direction = (1, 0, 0)
    elif axis == "y":
        direction = (0, 1, 0)
    elif axis == "z":
        direction = (0, 0, 1)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    result = gmsh.model.occ.revolve(
        [(2, surface)],
        0, 0, 0,                 
        *direction,              
        np.radians(angle)      
    )

    val = [tag for dim, tag in result if dim == 3]
    return val[0]

def subSurf(surf1, surf2):
    result, _ = gmsh.model.occ.cut(
        [(2, surf2)],
        [(2, surf1)]
    )

    gmsh.model.occ.synchronize()

    surfaces = [tag for dim, tag in result if dim == 2]
    if len(surfaces)==1:
        return surfaces[0]
    else:
        return surfaces

def uniteSurf(*surfaces):
    objects = [(2, tag) for tag in surfaces]

    result, _ = gmsh.model.occ.fuse(
        objects[0:1],
        objects[1:]
    )


    return [tag for dim, tag in result if dim == 2][0]

def make_taper(p1_tag,p2_tag,n_points=30,k=0):
    gmsh.model.occ.synchronize()
    p1=gmsh.model.getValue(0, p1_tag, [])
    p2=gmsh.model.getValue(0, p2_tag, [])
    t = np.linspace(0, 1, n_points)

    # y changes linearly
    y = p1[1] + t * (p2[1] - p1[1])

    # nonlinear y
    f = (1 + k) * t / (1 + k * t)
    x = p1[0] + f * (p2[0] - p1[0])

    # z
    z = p1[2] + t * (p2[2] - p1[2])

    coords = np.column_stack((x, y, z))

    ds = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate(([0], np.cumsum(ds)))
    s_equal = np.linspace(0, s[-1], n_points)

    coords = np.column_stack([
        np.interp(s_equal, s, coords[:, i])
        for i in range(3)
    ])

    point_tags = [p1_tag]

    for xyz in coords[1:-1]:
        point_tags.append(gmsh.model.occ.addPoint(*xyz))

    point_tags.append(p2_tag)

    return point_tags,coords


def genRecZ(width, height):
    """
    Create a rectangle centered at the origin in the xy-plane.
    Returns the surface tag.
    """
    return gmsh.model.occ.addRectangle(
        -width/2,
        -height/2,
        0,
        width,
        height
    )

def genRecY(width, height):
    """
    Create a rectangle centered at the origin in the xy-plane.
    Returns the surface tag.
    """

    rect = gmsh.model.occ.addRectangle(
        -width / 2,   # x
        -height / 2,  # y (will become z after rotation)
        0,
        width,
        height
    )

    # Rotate XY plane → XZ plane
    gmsh.model.occ.rotate(
        [(2, rect)],
        0, 0, 0,      # rotation center
        1, 0, 0,      # rotation axis
        -np.pi / 2
    )
    return rect



def movSurf(surf, dx, dy, dz):
    """
    Move a surface.
    """
    gmsh.model.occ.translate([(2, surf)], dx, dy, dz)

def movVol(vol, dx, dy, dz):
    """
    Move a volume.
    """
    gmsh.model.occ.translate([(3, vol)], dx, dy, dz)

def genBrick(x, y, z):
    """
    Create a brick centered at the origin.
    Returns the volume tag.
    """
    return gmsh.model.occ.addBox(
        -x/2,
        -y/2,
        -z/2,
        x,
        y,
        z
    )

def uniteVol(*vols):
    """
    Fuse multiple volumes into one.
    Returns the new volume tag.
    """
    objs = [(3, v) for v in vols]

    out, _ = gmsh.model.occ.fuse(
        objs[:1],
        objs[1:]
    )

    return out[0][1]

def subVol(tool, target):
    """
    Subtract tool volume from target volume.
    Returns the resulting volume tag.
    """
    result, _ = gmsh.model.occ.cut(
        [(3, target)],   # object to keep
        [(3, tool)]      # object to subtract
    )

    return result[0][1]

def delete_volume(volume_tag):
    gmsh.model.occ.remove([(3, volume_tag)], recursive=True)
    gmsh.model.occ.synchronize()

def subVol_keep(tool, target):
    """
    Subtract tool from target but keep the tool.
    Returns the resulting target volume tag.
    """
    result, _ = gmsh.model.occ.cut(
        [(3, target)],
        [(3, tool)],
        removeObject=True,
        removeTool=False   # <-- keep the tool
    )

    return result[0][1]

def subSurf_keep(tool, target):
    """
    Subtract tool from target but keep the tool.
    Returns the resulting target volume tag.
    """
    result, _ = gmsh.model.occ.cut(
        [(2, target)],
        [(2, tool)],
        removeObject=True,
        removeTool=False   # <-- keep the tool
    )

    return result[0][1]

def copySurf(surf):
    """
    Copy a surface.
    Returns the new surface tag.
    """
    copied = gmsh.model.occ.copy([(2, surf)])
    return copied[0][1]


def thickenSurf(surf, thickness):
    """
    Extrude a surface into a volume.
    Returns the new volume tag.
    """
    out = gmsh.model.occ.extrude(
        [(2, surf)],
        0,
        0,
        thickness
    )

    # Extrude returns several entities:
    # surface + lateral surfaces + volume
    vols = [tag for dim, tag in out if dim == 3]

    return vols[0]

def getSurfaces(vol):
    gmsh.model.occ.synchronize()
    return [tag for dim, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False)]

def getSurfaceCenter(surf):
    gmsh.model.occ.synchronize()
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf)
    return ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)

def getSurfaceEdges(surf):
    return [tag for dim, tag in gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False)]

def getCurveCenter(curve):
    return gmsh.model.occ.getCenterOfMass(1, curve)

def getEdges(vol):
    edges = set()
    for surf in getSurfaces(vol):
        edges.update(getSurfaceEdges(surf))
    return list(edges)


def getEdgeVertices(edge):
    return [t for d, t in gmsh.model.getBoundary([(1, edge)], oriented=False) if d == 0]

def getVertexPos(v):
    return np.array(gmsh.model.getValue(0, v, []))

def getBottomSurface(vol):
    """
    Get the surface of a volume with the lowest z-center.
    Returns surface tag.
    """

    gmsh.model.occ.synchronize()

    faces = gmsh.model.getBoundary(
        [(3, vol)],
        oriented=False,
        recursive=False
    )

    min_zcenter = float("inf")
    bottom = None

    for dim, tag in faces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
            dim,
            tag
        )
        zcenter = 0.5*(zmin + zmax)
        if zcenter < min_zcenter:
            min_zcenter = zcenter
            bottom = tag

    return bottom


def intersectVol_keep(vol1, vol2):
    """
    Intersect two volumes while keeping the originals.
    Returns the intersection volume.
    """

    result, _ = gmsh.model.occ.intersect(
        [(3, vol1)],
        [(3, vol2)],
        removeObject=False,
        removeTool=False
    )

    return result[0][1]

def getSurfaceNormal(surf):
    pts = []

    for edge in getSurfaceEdges(surf):
        for v in getEdgeVertices(edge):
            pts.append(getVertexPos(v))

    pts = np.unique(np.round(np.array(pts), 12), axis=0)

    p0 = pts[0]

    for i in range(1, len(pts)):
        for j in range(i + 1, len(pts)):
            v1 = pts[i] - p0
            v2 = pts[j] - p0

            n = np.cross(v1, v2)

            if np.linalg.norm(n) > 1e-10:
                return n / np.linalg.norm(n)

    raise RuntimeError("Couldn't determine surface normal.")

# def getSingularEdges(vol, angleTol=5):
#     singEdges = []

#     surfs = getSurfaces(vol)

#     edge2surf = {}

#     for surf in surfs:
#         for edge in getSurfaceEdges(surf):
#             edge2surf.setdefault(edge, []).append(surf)

#     for edge, adj in edge2surf.items():

#         if len(adj) != 2:
#             continue

#         n1 = getSurfaceNormal(adj[0])
#         n2 = getSurfaceNormal(adj[1])

#         theta = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1, 1)))

#         if abs(theta - 90) < angleTol:
#             singEdges.append(edge)

#     return singEdges

def colorVolumes(vols, alpha=150):
    gmsh.model.occ.synchronize()
    """
    Assign different colors to a list of volumes.

    Parameters
    ----------
    vols : list
        List of volume tags, e.g. [PadR, PadL, Gnd]
    alpha : int
        Transparency (0 = invisible, 255 = opaque)
    """

    # RGB colors to cycle through
    colors = [
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 0, 255),      # blue
        (255, 255, 0),    # yellow
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
        (255, 128, 0),    # orange
        (128, 0, 255),    # purple
        (128, 128, 128),  # gray
    ]

    for i, vol in enumerate(vols):
        r, g, b = colors[i % len(colors)]

        gmsh.model.setColor(
            [(3, vol)],
            r, g, b,
            alpha
        )

def get_edge_endpoints(edge_tag):
    pts = gmsh.model.getBoundary([(1, edge_tag)], oriented=False)
    return [p[1] for p in pts]


def is_vertical_edge(edge_tag, tol=1e-8):
    pts = get_edge_endpoints(edge_tag)

    xyz = []
    for p in pts:
        xyz.append(gmsh.model.getValue(0, p, []))

    p1 = np.array(xyz[0])
    p2 = np.array(xyz[1])

    # vertical if x,y do not change
    return abs(p1[0]-p2[0]) < tol and abs(p1[1]-p2[1]) < tol

def is_horizontolY_edge(edge_tag, tol=1e-8):
    pts = get_edge_endpoints(edge_tag)

    xyz = []
    for p in pts:
        xyz.append(gmsh.model.getValue(0, p, []))

    p1 = np.array(xyz[0])
    p2 = np.array(xyz[1])

    # vertical if x,y do not change
    return abs(p1[0]-p2[0]) < tol and abs(p1[2]-p2[2]) < tol

def getSurfaceVertices(surfTag):
    verts = gmsh.model.getBoundary([(2, surfTag)], oriented=False, recursive=True)
    return [tag for dim, tag in verts if dim == 0]

def splitCurveAtFraction(curveTag, f):
    pts = [tag for dim, tag in gmsh.model.getBoundary([(1, curveTag)], oriented=False)]

    x1 = gmsh.model.getValue(0, pts[0], [])
    x2 = gmsh.model.getValue(0, pts[1], [])

    xp = [(1-f)*x1[i] + f*x2[i] for i in range(3)]

    newPt = gmsh.model.occ.addPoint(*xp)

    out, _ = gmsh.model.occ.fragment([(1, curveTag)], [(0, newPt)])
    gmsh.model.occ.synchronize()

    newCurves = [tag for dim, tag in out if dim == 1]

    return newCurves

def get_top_surface(vol):

    surfaces = gmsh.model.getBoundary([(3, vol)], oriented=False)

    topSurf = None
    topZ = -float("inf")

    for _, surf in surfaces:
        z = gmsh.model.occ.getCenterOfMass(2, surf)[2]

        if z > topZ:
            topZ = z
            topSurf = surf

    return topSurf, topZ

def get_points_center(point_tags):
    coords = np.array([
        gmsh.model.getValue(0, tag, [])
        for tag in point_tags
    ])

    return coords.mean(axis=0)

def get_rightmost_points(surface_tag, tol=1e-9):
    # Get curves bounding the surface
    curves=getSurfaceEdges(surface_tag)

    points=[]
    for ii in curves:
        a,b=get_edge_endpoints(ii)
        if a not in points:
            points.append(a)
        if b not in points:
            points.append(b)

    # Get coordinates
    point_data = []
    for tag in points:
        xyz = gmsh.model.getValue(0, tag, [])
        point_data.append((tag, xyz))

    # Maximum x
    xmax = max(xyz[0] for _, xyz in point_data)

    # All points at maximum x
    rightmost = [
        (tag, xyz)
        for tag, xyz in point_data
        if abs(xyz[0] - xmax) < tol
    ]

    return rightmost

def get_bottom_points(surface_tag, tol=1e-9):
    # Get curves bounding the surface
    curves=getSurfaceEdges(surface_tag)

    points=[]
    for ii in curves:
        a,b=get_edge_endpoints(ii)
        if a not in points:
            points.append(a)
        if b not in points:
            points.append(b)

    # Get coordinates
    point_data = []
    for tag in points:
        xyz = gmsh.model.getValue(0, tag, [])
        point_data.append((tag, xyz))

    # min y
    ymin = min(xyz[1] for _, xyz in point_data)

    # All points at min y
    botmost = [
        (tag, xyz)
        for tag, xyz in point_data
        if abs(xyz[1] - ymin) < tol
    ]

    return botmost

def get_top_points(surface_tag, tol=1e-9):
    # Get curves bounding the surface
    curves=getSurfaceEdges(surface_tag)

    points=[]
    for ii in curves:
        a,b=get_edge_endpoints(ii)
        if a not in points:
            points.append(a)
        if b not in points:
            points.append(b)

    # Get coordinates
    point_data = []
    for tag in points:
        xyz = gmsh.model.getValue(0, tag, [])
        point_data.append((tag, xyz))

    # max y
    ymin = max(xyz[1] for _, xyz in point_data)

    # All points at max y
    topmost = [
        (tag, xyz)
        for tag, xyz in point_data
        if abs(xyz[1] - ymin) < tol
    ]

    return topmost


def get_leftmost_points(surface_tag, tol=1e-9):
    # Get curves bounding the surface
    curves=getSurfaceEdges(surface_tag)

    points=[]
    for ii in curves:
        a,b=get_edge_endpoints(ii)
        if a not in points:
            points.append(a)
        if b not in points:
            points.append(b)

    # Get coordinates
    point_data = []
    for tag in points:
        xyz = gmsh.model.getValue(0, tag, [])
        point_data.append((tag, xyz))

    # Maximum x
    xmax = min(xyz[0] for _, xyz in point_data)

    # All points at maximum x
    leftmost = [
        (tag, xyz)
        for tag, xyz in point_data
        if abs(xyz[0] - xmax) < tol
    ]

    return leftmost


def make_cylinders_from_vertical_edges(edges, radius):

    cylinders = []
    height_val =[]
    edge_val=[]

    for edge in edges:

        if not is_vertical_edge(edge):
            continue

        pts = get_edge_endpoints(edge)

        p1 = np.array(gmsh.model.getValue(0, pts[0], []))
        p2 = np.array(gmsh.model.getValue(0, pts[1], []))

        height = np.linalg.norm(p2-p1)

        # start point of cylinder
        x, y = p1[0], p1[1]
        z = min(p1[2], p2[2])

        cyl = gmsh.model.occ.addCylinder(
            x, y, z,
            0, 0, height,
            radius
        )

        cylinders.append(cyl)
        height_val.append(height)
        edge_val.append(edge)


    return cylinders,height_val,edge_val

def count_triangles_in_physical_group(group_name):
    """
    Return the number of triangular elements in a 2D physical group.

    Parameters
    ----------
    group_name : str
        Name of the 2D physical group.

    Returns
    -------
    int
        Number of triangle elements.
    """
    # Find the physical group tag
    phys_tag = None
    for dim, tag in gmsh.model.getPhysicalGroups(dim=2):
        if gmsh.model.getPhysicalName(dim, tag) == group_name:
            phys_tag = tag
            break

    if phys_tag is None:
        raise ValueError(f"Physical group '{group_name}' not found.")

    ntri = 0
    surfaces = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)

    for surf in surfaces:
        elemTypes, elemTags, _ = gmsh.model.mesh.getElements(2, surf)

        for etype, tags in zip(elemTypes, elemTags):
            name, dim, *_ = gmsh.model.mesh.getElementProperties(etype)
            if dim == 2 and "Triangle" in name:
                ntri += len(tags)

    return ntri

def count_tetrahedrons_in_volume(volume_tag):
    """
    Count tetrahedral elements in a Gmsh volume.

    Parameters
    ----------
    volume_tag : int
        Volume entity tag.

    Returns
    -------
    int
        Number of tetrahedral elements.
    """
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(3, volume_tag)

    ntets = 0

    for etype, tags in zip(elemTypes, elemTags):
        name, dim, *_ = gmsh.model.mesh.getElementProperties(etype)

        if dim == 3 and "Tetrahedron" in name:
            ntets += len(tags)

    return ntets

def count_tetrahedrons_in_physical_volume(group_name):
    """
    Count tetrahedral elements in a 3D physical group by name.
    """
    phys_tag = None

    for dim, tag in gmsh.model.getPhysicalGroups(dim=3):
        if gmsh.model.getPhysicalName(dim, tag) == group_name:
            phys_tag = tag
            break

    if phys_tag is None:
        raise ValueError(f"Physical volume '{group_name}' not found")

    ntets = 0
    volumes = gmsh.model.getEntitiesForPhysicalGroup(3, phys_tag)

    for vol in volumes:
        ntets += count_tetrahedrons_in_volume(vol)

    return ntets