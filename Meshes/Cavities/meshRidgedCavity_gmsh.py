import gmsh
import numpy as np
import sys
sys.path.append("./Meshes/")
from Cubit2Gmsh import *

gmsh.initialize()
gmsh.model.add("model")

## Geometry definitions
cav_length = 0.01
cav_width = 0.005
cav_height = 0.0075
disc_height = cav_height / 10 # Height of the discontinuity

## Draw the geometry
# Create the cavity itself
cav_vol=genBrick(cav_width,cav_length,cav_height)

# Create the rectangular discontinuity
disc_vol=genBrick(cav_width,cav_length/2,disc_height)
movVol(disc_vol, 0, -cav_length/4, 0)

# Subtract the discontinuity from the cavity
subVol(disc_vol, cav_vol)
cav_surfs=getSurfaces(cav_vol)
cav_curves=getEdges(cav_vol) # All edges in the volume

## Isolate the singular edges
singLineCenters = np.array([[0, 0, -disc_height/2], [0, 0, disc_height/2]], dtype=np.float64)
singLines = []
for curve in cav_curves:
    curve_center = getCurveCenter(curve)
    for singLineCenter in singLineCenters:
        if np.all(np.isclose(curve_center, singLineCenter, atol=1e-4)):
            singLines.append(curve)

## Set mesh settings
target_size = 0.001 # Target for the average edge length of the tets
gmsh.option.setNumber("Mesh.MeshSizeMin", target_size)
gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)

## Generate the mesh
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.generate(3)


## Make Blocks
blockCnt = 1
phys=gmsh.model.addPhysicalGroup(3, [cav_vol], blockCnt) # Block for the cavity volume
gmsh.model.setPhysicalName(3, phys, "mpr1epr1") # 3 is the dimension (3D volume), mpr1epr1 is the name

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2, cav_surfs, tag=blockCnt) # Outer boundary block
gmsh.model.setPhysicalName(2, phys, "Outer") # 2 is the dimension (2D surfaces), Outer is the name

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(1, singLines, tag=blockCnt) # Singular line block
gmsh.model.setPhysicalName(1, phys, "singLines") # 1 is the dimension (1D edges), singLines is the name

## Print number of Tets
types, elemTags, nodeTags = gmsh.model.mesh.getElements(3)
tet_count = 0
for t, tags in zip(types, elemTags):
    if t == 4:   # 4-node tetrahedron
        tet_count += len(tags)

print("Number of tetrahedrons:", tet_count)
gmsh.write(f"./Meshes/Cavities/ridgedCavity.msh")
gmsh.model.occ.synchronize()
# gmsh.write("microstrip_gmsh_coarse.msh")
# gmsh.write("microstrip_gmsh.brep")
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()