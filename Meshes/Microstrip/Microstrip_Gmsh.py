import gmsh
import numpy as np
import sys
from Cubit2Gmsh import *
from collections import Counter

gmsh.initialize()
gmsh.model.add("model")

############## Geometry definitions
# Substrate
sub_width = 5e-3
sub_height = 8e-4
sub_length = 10e-3
sub_er = 7

# Signal line
sig_width = sub_width / 5
sig_height = sub_height / 20
sig_length = sub_length

# Ground plane
gnd_width = sub_width
#gnd_height = sig_height*3
gnd_length = sub_length

# Bounding box
box_width = sub_width*2
box_height = sub_height*10
box_length = sub_length*2

# Lumped ports
port_width = sig_width
port_height = sub_height

########## Start Making
sub_vol=genBrick(sub_width,sub_length,sub_height)
sub_surfs=getSurfaces(sub_vol)

sig_vol=genRecZ(sig_width, sig_length)
movSurf(sig_vol, 0, 0,sub_height/2)

port1_vol=genRecY(port_width, port_height)
movSurf(port1_vol, 0, -sub_length/2,0)

port2_vol=genRecY(port_width, port_height)
movSurf(port2_vol, 0, +sub_length/2,0)

gnd_vol=getBottomSurface(sub_vol)

box_vol=genBrick(box_width, box_length, box_height)

entities=gmsh.model.occ.getEntities()
gmsh.model.occ.rotate(
    entities,
    0,0,0,
    1,0,0,
    -np.pi/2

)

imp_surfaces =[sig_vol,port1_vol,port2_vol,gnd_vol]
imp_volumes = [ sub_vol,box_vol]
############ Make Conformal ####################
gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()
surfaces = gmsh.model.occ.getEntities(dim=2)
volumes = gmsh.model.occ.getEntities(dim=3)
out, out_map =gmsh.model.occ.fragment(surfaces, surfaces)
imp_surfaces=updateSurfList(imp_surfaces,surfaces+surfaces,out_map)
gmsh.model.occ.synchronize()
surfaces = gmsh.model.occ.getEntities(dim=2)
volumes = gmsh.model.occ.getEntities(dim=3)
out, out_map =gmsh.model.occ.fragment(volumes, surfaces)
imp_surfaces=updateSurfList(imp_surfaces,volumes+surfaces,out_map)
gmsh.model.occ.synchronize()
surfaces = gmsh.model.occ.getEntities(dim=2)
volumes = gmsh.model.occ.getEntities(dim=3)
out, out_map =gmsh.model.occ.fragment(volumes, volumes)
imp_volumes = updateVolList(imp_volumes,volumes+volumes,out_map)
gmsh.model.occ.synchronize()
 
sig_vol,port1_vol,port2_vol,gnd_vol=imp_surfaces
sub_vol,box_vol = imp_volumes



## Meshing

distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", [sig_vol])
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
gmsh.model.mesh.field.setNumber(threshold, "SizeMin", 0.001)
gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 0.6)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", 1.5)

field_math=gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(field_math, "F", "0.0004")

restrict=gmsh.model.mesh.field.add("Restrict")
gmsh.model.mesh.field.setNumber(restrict, "InField", field_math)
gmsh.model.mesh.field.setNumbers(
    restrict, "SurfacesList", [port1_vol,port2_vol]
)

Min=gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(
    Min, "FieldsList", [threshold, restrict]
)
gmsh.model.mesh.field.setAsBackgroundMesh(Min)

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.generate(3)


## Make Blocks
blockCnt = 1
phys=gmsh.model.addPhysicalGroup(3, [sub_vol], blockCnt)
gmsh.model.setPhysicalName(3, phys, "mpr1epr10")

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(3, [box_vol], blockCnt)
gmsh.model.setPhysicalName(3, phys, "mpr1epr1")

vols = gmsh.model.getEntities(3)
all_surfs = []
for vol in vols:
    bnd = gmsh.model.getBoundary([vol], oriented=False, recursive=False)
    all_surfs.extend(tag for dim, tag in bnd if dim == 2)
counts = Counter(all_surfs)
outer_surfs = [s for s, c in counts.items() if c == 1]
blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2, outer_surfs, tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "Outer")

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2,[sig_vol], tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "PEC1")

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2,[gnd_vol], tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "PEC2")

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2,[port1_vol], tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "Port1")

blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2,[port2_vol], tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "Port2")

## Print number of Tets
types, elemTags, nodeTags = gmsh.model.mesh.getElements(3)
tet_count = 0
for t, tags in zip(types, elemTags):
    if t == 4:   # 4-node tetrahedron
        tet_count += len(tags)

print("Number of tetrahedrons:", tet_count)

gmsh.model.occ.synchronize()
# gmsh.write("microstrip_gmsh_coarse.msh")
# gmsh.write("microstrip_gmsh.brep")
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()