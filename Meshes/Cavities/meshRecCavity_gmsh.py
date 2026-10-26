import gmsh
import numpy as np
import sys
sys.path.append("./Meshes/")
from Cubit2Gmsh import *

gmsh.initialize()
gmsh.model.add("model")

## Geometry definitions
cav_width = 0.01
cav_length = 0.0075
cav_height = 0.005

## Draw the cavity
cav_vol=genBrick(cav_width,cav_length,cav_height)
cav_surfs=getSurfaces(cav_vol)

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
gmsh.model.setPhysicalName(2, phys, "Outer") # 2 is the dimension (2D surfaces), mpr1epr1 is the name

## Print number of Tets
types, elemTags, nodeTags = gmsh.model.mesh.getElements(3)
tet_count = 0
for t, tags in zip(types, elemTags):
    if t == 4:   # 4-node tetrahedron
        tet_count += len(tags)

print("Number of tetrahedrons:", tet_count)
gmsh.write(f"./Meshes/Cavities/recCavity_TEST.msh")
gmsh.model.occ.synchronize()
# gmsh.write("microstrip_gmsh_coarse.msh")
# gmsh.write("microstrip_gmsh.brep")
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()