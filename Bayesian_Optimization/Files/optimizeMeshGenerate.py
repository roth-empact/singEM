import gmsh
import numpy as np
import meshio
import sys
from Cubit2Gmsh import *
from collections import Counter

gmsh.initialize()
gmsh.model.add("model")

#### Numbers to vary ####
PadX=float(sys.argv[1])*1e-3
PadY=float(sys.argv[2])*1e-3
Junc_len=float(sys.argv[3])*1e-3

Pad2GndY_U=float(sys.argv[4])*1e-3
Pad2GndY_D=float(sys.argv[4])*1e-3

Pad2GndX_L=float(sys.argv[5])*1e-3
Pad2GndX_R=float(sys.argv[5])*1e-3

trenchZ=float(sys.argv[6])*1e-3
PJunc_W=float(sys.argv[7])*1e-3
k_val=float(sys.argv[8])
dir_GK=sys.argv[9]
iter_GK=sys.argv[10]


#### constant for now####
n_val=200
Full_Gnd=3000e-3

#### Numbers that don't vary ####
PortX=0.002
PortY=0.002923
lenX=1e3*(2*PadX+2*Junc_len+Pad2GndX_L+Pad2GndX_R+PortX)
lenY=1e3*(PadY+Pad2GndY_U+Pad2GndX_L)
metal_thick=0.0001
metalHeight=metal_thick ## defined twice cause copying some Sam's mesh code
botZPos=-metal_thick/2
ParaPadX_s=0.08
ParaPadX_e=0.12
ParaPadY_s=0.2
ParaPadY_e=0.6
SubZ=0.3+metal_thick
Trench_Junc_GapX=PortX+2*Junc_len
Trench_Junc_GapY=0.02
msZ=0.01 
ExCubeXY=0.01
ExCubeZ=0.01+metal_thick
AirBoxZ=0.5-metal_thick

## Some shifts that were defined to match the paper IBM geometry but not necessary for optimization
ShiftX=0#0.007142
Junc_ShiftY=0#0.0014615
ShiftY=0#-PadY/2+100e-3#-0.033871

### Make Port
GapX = 2*PadX + PortX + 2*Junc_len + Pad2GndX_L+Pad2GndX_R
GapY = PadY + Pad2GndY_U + Pad2GndY_D
GndX = Full_Gnd
GndY = Full_Gnd
Port = genRecZ(PortX, PortY)
movSurf(Port, 0, 0,metal_thick/2)
movSurf(Port, 0, Junc_ShiftY+ShiftY,0)

## Make dummy Port
Dummy_Port=genBrick(PortX,PortY,metal_thick)
movVol(Dummy_Port,0,Junc_ShiftY+ShiftY,0)

## Make Left Pad
JuncL=genBrick(ExCubeXY/2-PortX/2,PortY,metal_thick)
movVol(JuncL,-(ExCubeXY/2-PortX/2)/2-PortX/2,+Junc_ShiftY+ShiftY,0)
PadL=genBrick(PadX,PadY,metal_thick)
movVol(PadL,-Junc_len-PortX/2-PadX/2,0,0)
p1 = gmsh.model.occ.addPoint(-ExCubeXY/2,-PortY/2+ShiftY,metal_thick/2)
p2 = gmsh.model.occ.addPoint(-ExCubeXY/2,+PortY/2+ShiftY,metal_thick/2)
p3 = gmsh.model.occ.addPoint(-PortX/2-Junc_len,+PJunc_W/2+ShiftY,metal_thick/2)
p4 = gmsh.model.occ.addPoint(-PortX/2-Junc_len,-PJunc_W/2+ShiftY,metal_thick/2)
curve_points_botL,coords_botL=make_taper(p1,p4,n_val,k_val)
curve_points_topL,coords_topL=make_taper(p2,p3,n_val,k_val)
l1_L=gmsh.model.occ.addLine(curve_points_topL[0],curve_points_botL[0])
l2_L=gmsh.model.occ.addSpline(curve_points_botL)
l3_L=gmsh.model.occ.addLine(curve_points_topL[-1],curve_points_botL[-1])
l4_L=gmsh.model.occ.addSpline(curve_points_topL)
loop_L = gmsh.model.occ.addCurveLoop([l1_L, l2_L, l3_L, l4_L])
trapL = gmsh.model.occ.addPlaneSurface([loop_L])
trapL=thickenSurf(trapL,-metal_thick)
PadL=uniteVol(trapL,PadL)
PadL=uniteVol(JuncL,PadL)

## Make Right Pad
JuncR=genBrick(ExCubeXY/2-PortX/2,PortY,metal_thick)
movVol(JuncR,+(ExCubeXY/2-PortX/2)/2+PortX/2,+Junc_ShiftY+ShiftY,0)
PadR=genBrick(PadX,PadY,metal_thick)
movVol(PadR,+Junc_len+PortX/2+PadX/2,0,0)
p1 = gmsh.model.occ.addPoint(+ExCubeXY/2,-PortY/2+ShiftY,metal_thick/2)
p2 = gmsh.model.occ.addPoint(+ExCubeXY/2,+PortY/2+ShiftY,metal_thick/2)
p3 = gmsh.model.occ.addPoint(+PortX/2+Junc_len,+PJunc_W/2+ShiftY,metal_thick/2)
p4 = gmsh.model.occ.addPoint(+PortX/2+Junc_len,-PJunc_W/2+ShiftY,metal_thick/2)
curve_points_botR,coords_botR=make_taper(p1,p4,n_val,k_val)
curve_points_topR,coords_topR=make_taper(p2,p3,n_val,k_val)
l1_R=gmsh.model.occ.addLine(curve_points_topR[0],curve_points_botR[0])
l2_R=gmsh.model.occ.addSpline(curve_points_botR)
l3_R=gmsh.model.occ.addLine(curve_points_topR[-1],curve_points_botR[-1])
l4_R=gmsh.model.occ.addSpline(curve_points_topR)
loop_R = gmsh.model.occ.addCurveLoop([l1_R, l2_R, l3_R, l4_R])
trapR = gmsh.model.occ.addPlaneSurface([loop_R])
trapR=thickenSurf(trapR,-metal_thick)
PadR=uniteVol(trapR,PadR)
PadR=uniteVol(JuncR,PadR)


## Make Gnd
Gnd = genBrick(GndX, GndY, metal_thick)

Gap = genBrick(GapX, GapY, metal_thick)
movVol(
    Gap,
    GapX/2-PortX/2-Junc_len-PadX-Pad2GndX_L,
    GapY/2-PadY/2-Pad2GndY_D,
    0
)
Gnd = subVol(Gap, Gnd)

## Make Air Around Pads
Air_Around_Pads = genBrick(GapX, GapY, metal_thick)
movVol(
    Air_Around_Pads,
    GapX/2-PortX/2-Junc_len-PadX-Pad2GndX_L,
    GapY/2-PadY/2-Pad2GndY_D,
    0
)
for vol in [PadR, PadL, Dummy_Port]:
    Air_Around_Pads = subVol_keep(vol, Air_Around_Pads)

## Make PadR MS
bottom = getBottomSurface(PadR)
bottom_copy = copySurf(bottom)
PadR_MS = thickenSurf(bottom_copy, -msZ)

## Make PadL MS
bottom = getBottomSurface(PadL)
bottom_copy = copySurf(bottom)
PadL_MS = thickenSurf(bottom_copy, -msZ)

## Make Gnd MS
bottom = getBottomSurface(Gnd)
bottom_copy = copySurf(bottom)
Gnd_MS = thickenSurf(bottom_copy, -msZ)

## Make Trench
if trenchZ!=0:
    bottom = getBottomSurface(Air_Around_Pads)
    bottom_copy = copySurf(bottom)
    Trench = thickenSurf(bottom_copy, -trenchZ)
    Tgap_topL=[gmsh.model.occ.addPoint(*p) for p in list(coords_topL+np.tile([0,Trench_Junc_GapY/2,-metal_thick],(n_val,1)))]
    Tgap_topR=[gmsh.model.occ.addPoint(*p) for p in list(coords_topR+np.tile([0,Trench_Junc_GapY/2,-metal_thick],(n_val,1)))]
    Tgap_botL=[gmsh.model.occ.addPoint(*p) for p in list(coords_botL+np.tile([0,-Trench_Junc_GapY/2,-metal_thick],(n_val,1)))]
    Tgap_botR=[gmsh.model.occ.addPoint(*p) for p in list(coords_botR+np.tile([0,-Trench_Junc_GapY/2,-metal_thick],(n_val,1)))]
    l1_T=gmsh.model.occ.addSpline(Tgap_topL)
    l2_T=gmsh.model.occ.addLine(Tgap_topL[0],Tgap_topR[0])
    l3_T=gmsh.model.occ.addSpline(Tgap_topR)
    l4_T=gmsh.model.occ.addLine(Tgap_topR[-1],Tgap_botR[-1])
    l5_T=gmsh.model.occ.addSpline(Tgap_botR)
    l6_T=gmsh.model.occ.addLine(Tgap_botR[0],Tgap_botL[0])
    l7_T=gmsh.model.occ.addSpline(Tgap_botL)
    l8_T=gmsh.model.occ.addLine(Tgap_botL[-1],Tgap_topL[-1])
    loop = gmsh.model.occ.addCurveLoop([l1_T, l2_T, l3_T, l4_T,l5_T,l6_T,l7_T,l8_T])
    loop2 = gmsh.model.occ.addPlaneSurface([loop])
    Trench_Gap=thickenSurf(loop2,-trenchZ)
    Trench = subVol(Trench_Gap,Trench)
else:
    Trench=None

## Make Exclusion Box
ExCube = genBrick(ExCubeXY,ExCubeXY,ExCubeZ)
movVol(ExCube,0,0,-ExCubeZ/2 + metal_thick/2)
movVol(ExCube,0,Junc_ShiftY+ShiftY,0)
PadL_bit = intersectVol_keep(PadL,ExCube)
PadL = subVol_keep(PadL_bit,PadL)
PadR_bit = intersectVol_keep(PadR,ExCube)
PadR = subVol_keep(PadR_bit,PadR)
for vol in [PadR_bit, PadL_bit, PadR_MS, PadL_MS]:
    ExCube = subVol_keep(vol,ExCube)

## Make Sub
Sub = genBrick(GndX, GndY, SubZ)
movVol(Sub, 0, 0, -SubZ/2 + metal_thick/2)
for ii in [PadR_bit, PadL_bit, PadR, PadL, Gnd, Air_Around_Pads, PadR_MS, PadL_MS, Trench, Gnd_MS, ExCube]:
    if ii is not None:
        Sub = subVol_keep(ii, Sub)

## Subtract Metal Layer from ExCube
Metal_Gap = genBrick(Trench_Junc_GapX, Trench_Junc_GapY, metal_thick)
movVol(Metal_Gap, 0, Junc_ShiftY+ShiftY, 0)
ExCube = subVol(Metal_Gap, ExCube)

## Unite Dummy Port with ExCube
ExCube = uniteVol(ExCube, Dummy_Port)

## Make Top Air
Air = genBrick(GndX, GndY, AirBoxZ)
movVol(Air, 0, 0, AirBoxZ/2 + metal_thick/2)

## Imprint and merge
gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()
surfaces = gmsh.model.occ.getEntities(dim=2)
volumes = gmsh.model.occ.getEntities(dim=3)
gmsh.model.occ.fragment(surfaces, surfaces)
gmsh.model.occ.fragment(volumes, surfaces)
gmsh.model.occ.fragment(volumes, volumes)
gmsh.model.occ.synchronize()




## Redefining terms
bulkVol = Sub # Substrate volume
boxVol = Air # Bounding box volume
gndVol = Gnd # Ground plane volume
padVols = [PadL, PadR] # PEC part of pads
padSubVols=[PadL_MS,PadR_MS]
trenchVol=Trench
portSurf=Port
gapVol=Air_Around_Pads
gndSubVol=Gnd_MS
cutBoxVol=ExCube
padCutVols=[PadL_bit,PadR_bit]


# Get the interface surfaces
bulkSurfs = getSurfaces(bulkVol)
boxSurfs = getSurfaces(boxVol)
gndSurfs = getSurfaces(gndVol)
padSurfs = [getSurfaces(vol) for vol in padVols]
padSubSurfs = [getSurfaces(vol) for vol in padSubVols]
if trenchVol!=None:
    trenchSurfs = getSurfaces(trenchVol)
else:
    trenchSurfs=None
gapSurfs = getSurfaces(gapVol)
Gnd_MSSurfs = getSurfaces(Gnd_MS)
padMSSurfs = [[],[]]
for ii in range(2):
    for surf in padSurfs[ii]:
        if surf in padSubSurfs[ii]:
            padMSSurfs[ii]=surf
SASurfs = []
valid_surfaces=set(Gnd_MSSurfs) | set(padSubSurfs[0]) | set(padSubSurfs[1]) | set(bulkSurfs)
if trenchSurfs!=None:
    for surf in trenchSurfs:
        if surf in valid_surfaces:
            SASurfs.append(surf)
    SASurfs=SASurfs+list(set(bulkSurfs) & set(gapSurfs))
else:
    SASurfs=list(set(bulkSurfs) & set(gapSurfs))

padMASideSurfs = [[], []]
padMATopSurfs = [0, 0]
gndMASideSurfs = []
for ii in range(2):
    for surf in padSurfs[ii]:
        if surf in boxSurfs:
            padMATopSurfs[ii]=surf
        if surf in gapSurfs:
            padMASideSurfs[ii].append(surf)

for surf in gndSurfs:
    if surf in gapSurfs:
        gndMASideSurfs.append(surf)

gndMATopSurf=0
for surf in gndSurfs:
    if surf in boxSurfs:
        gndMATopSurf=surf

gndMSSurfs=[]
for surf in gndSurfs:
    if surf in Gnd_MSSurfs:
        gndMSSurfs.append(surf)

vols = gmsh.model.getEntities(3)

all_surfs = []
for vol in vols:
    bnd = gmsh.model.getBoundary([vol], oriented=False, recursive=False)
    all_surfs.extend(tag for dim, tag in bnd if dim == 2)

counts = Counter(all_surfs)
outer_surfs = [s for s, c in counts.items() if c == 1]

outer_edges=[]
for ii in outer_surfs:
    outer_edge_vals=getSurfaceEdges(ii)
    outer_edges.extend(outer_edge_vals)
outside_edges=list(set(outer_edges))
### Identify singular lines on PadR
rad=1e-3
PadR_sing=getEdges(PadR)
PadR_sing=[item for item in PadR_sing if item not in (getEdges(PadR_bit) + outside_edges)]
cyl_val,height_val,edge_val = make_cylinders_from_vertical_edges(PadR_sing, radius=rad)
for ii in range(len(cyl_val)):
    Cyl=subVol_keep(PadR,cyl_val[ii])
    if gmsh.model.occ.getMass(3, Cyl)<(height_val[ii]*rad**2*np.pi/2):
        gmsh.model.occ.remove([(3, Cyl)], recursive=True)
        PadR_sing.remove(edge_val[ii])
    else:
        gmsh.model.occ.remove([(3, Cyl)], recursive=True)
PadR_sing2=getEdges(PadR_bit)
PadR_sing2=[item for item in PadR_sing2 if not is_vertical_edge(item)]
PadR_sing2=[item for item in PadR_sing2 if not is_horizontolY_edge(item)]

portVertex=getSurfaceVertices(portSurf)
to_del=[]
to_add=[]
for e in PadR_sing2:
    pts=getEdgeVertices(e)
    if any(p in portVertex for p in pts):
        to_del.append(e)
        new_curves=splitCurveAtFraction(e,0.5)
        for e2 in new_curves:
            pts2=getEdgeVertices(e2)
            if any(p2 in portVertex for p2 in pts2):
                continue
            else:
                to_add.append(e2)
PadR_sing2=[item for item in PadR_sing2 if item not in to_del]+list(to_add)



### Identify singular lines on PadL
rad=1e-3
PadL_sing=getEdges(PadL)
PadL_sing=[item for item in PadL_sing if item not in (getEdges(PadL_bit) + outside_edges)]
cyl_val,height_val,edge_val = make_cylinders_from_vertical_edges(PadL_sing, radius=rad)
for ii in range(len(cyl_val)):
    Cyl=subVol_keep(PadL,cyl_val[ii])
    if gmsh.model.occ.getMass(3, Cyl)<(height_val[ii]*rad**2*np.pi/2):
        gmsh.model.occ.remove([(3, Cyl)], recursive=True)
        PadL_sing.remove(edge_val[ii])
    else:
        gmsh.model.occ.remove([(3, Cyl)], recursive=True)
PadL_sing2=getEdges(PadL_bit)
PadL_sing2=[item for item in PadL_sing2 if not is_vertical_edge(item)]
PadL_sing2=[item for item in PadL_sing2 if not is_horizontolY_edge(item)]

portVertex=getSurfaceVertices(portSurf)
to_del=[]
to_add=[]
for e in PadL_sing2:
    pts=getEdgeVertices(e)
    if any(p in portVertex for p in pts):
        to_del.append(e)
        new_curves=splitCurveAtFraction(e,0.5)
        for e2 in new_curves:
            pts2=getEdgeVertices(e2)
            if any(p2 in portVertex for p2 in pts2):
                continue
            else:
                to_add.append(e2)
PadL_sing2=[item for item in PadL_sing2 if item not in to_del]+list(to_add)


### Identify singular lines on Gnd
rad=1e-3
Gnd_sing=getEdges(Gnd)
Gnd_sing=[item for item in Gnd_sing if item not in outside_edges]
cyl_val,height_val,edge_val = make_cylinders_from_vertical_edges(Gnd_sing, radius=rad)
for ii in range(len(cyl_val)):
    Cyl=subVol_keep(Gnd,cyl_val[ii])
    if gmsh.model.occ.getMass(3, Cyl)<(height_val[ii]*rad**2*np.pi/2):
        gmsh.model.occ.remove([(3, Cyl)], recursive=True)
        Gnd_sing.remove(edge_val[ii])
    else:
        gmsh.model.occ.remove([(3, Cyl)], recursive=True)

singLines=Gnd_sing+PadL_sing+PadR_sing+PadR_sing2+PadL_sing2

# ## find top and bottom of geometry
# surfaces = gmsh.model.getEntities(2)   # List of (2, tag)
# topSurf = None
# botSurf = None
# topZ = -float("inf")
# botZ = float("inf")
# for _, tag in surfaces:
#     _, _, z = gmsh.model.occ.getCenterOfMass(2, tag)

#     if z > topZ:
#         topZ = z
#         topSurf = tag

#     if z < botZ:
#         botZ = z
#         botSurf = tag

### Begin Meshing
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", [portSurf])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "InField", distance)

gmsh.model.mesh.field.setNumber(threshold, "SizeMin", 0.001)
gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 0.15)

# Distance over which the transition occurs
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.8)

gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.generate(3)

## Print number of Tets
types, elemTags, nodeTags = gmsh.model.mesh.getElements(3)
tet_count = 0
for t, tags in zip(types, elemTags):
    if t == 4:   # 4-node tetrahedron
        tet_count += len(tags)

print("Number of tetrahedrons:", tet_count)


# ### Physical Groups
blockCnt = 1
phys=gmsh.model.addPhysicalGroup(3, [bulkVol, gndSubVol, padSubVols[0], padSubVols[1], cutBoxVol], tag=blockCnt)
gmsh.model.setPhysicalName(3, phys, "mpr1epr10")
blockCnt += 1
if trenchVol!=None:
    phys=gmsh.model.addPhysicalGroup(3, [boxVol, gapVol, trenchVol], tag=blockCnt)
else:
    phys=gmsh.model.addPhysicalGroup(3, [boxVol, gapVol], tag=blockCnt)
gmsh.model.setPhysicalName(3, phys, "mpr1epr1")


gnd_check=[]
for ii in getSurfaces(Gnd):
    if ii not in outer_surfs:
        gnd_check.append(ii)
blockCnt += 1
phys=gmsh.model.addPhysicalGroup(2, gnd_check+outer_surfs, tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "PEC0")


blockCnt += 1  
pec1_surfs =list(set(getSurfaces(padCutVols[0])) ^ set(getSurfaces(padVols[0])))
phys=gmsh.model.addPhysicalGroup(2, pec1_surfs, tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "PEC1")

blockCnt += 1  
pec2_surfs =list(set(getSurfaces(padCutVols[1])) ^ set(getSurfaces(padVols[1])))
phys=gmsh.model.addPhysicalGroup(2, pec2_surfs, tag=blockCnt)
gmsh.model.setPhysicalName(2, phys, "PEC2")

blockCnt += 1 
phys = gmsh.model.addPhysicalGroup(2, [portSurf], blockCnt)
gmsh.model.setPhysicalName(2, phys, "Port1")

# Add metal air interfaces
surfSet = [gndMATopSurf, padMATopSurfs[0], padMATopSurfs[1]]
names = ["MATopGND", "MATopPad1", "MATopPad2"]
for ii, surf in enumerate(surfSet):
    blockCnt += 1
    phys = gmsh.model.addPhysicalGroup(2, [surf], blockCnt)
    gmsh.model.setPhysicalName(2, phys, names[ii])

# Add metal air side surfaces
surfSet = [gndMASideSurfs, padMASideSurfs[0],padMASideSurfs[1]]
names = ["MASideGND", "MASidePad1", "MASidePad2"]
for ii, surf in enumerate(surfSet):
    blockCnt += 1
    phys = gmsh.model.addPhysicalGroup(2, surf, blockCnt)
    gmsh.model.setPhysicalName(2, phys, names[ii])

# Add metal substrate interfaces
surfSet = [gndMSSurfs, [padMSSurfs[0]], [padMSSurfs[1]]]
names = ["MSGND", "MSPad1", "MSPad2"]
for ii, surf in enumerate(surfSet):
    blockCnt += 1
    phys = gmsh.model.addPhysicalGroup(2, surf, blockCnt)
    gmsh.model.setPhysicalName(2, phys, names[ii])

blockCnt += 1
phys = gmsh.model.addPhysicalGroup(2, SASurfs, blockCnt)
gmsh.model.setPhysicalName(2, phys, "SA")

blockCnt += 1
phys = gmsh.model.addPhysicalGroup(1, singLines, blockCnt)
gmsh.model.setPhysicalName(1, phys, "singLines")

gmsh.write(f"{dir_GK}/Gmsh_PadX_{(PadX*1e3):.3g}_PadY_{(PadY*1e3):.3g}_Junclen_{(Junc_len*1e3):.3g}_Pad2GndX_{(Pad2GndX_L*1e3):.3g}_Pad2GndY_{(Pad2GndY_U*1e3):.3g}_TaperW_{(PJunc_W*1e3):.3g}_TrenchZ_{(trenchZ*1e3):.3g}_Kval_{(k_val):.3g}_test_{iter_GK}.msh")

# if "-nopopup" not in sys.argv:
#     gmsh.fltk.run()

gmsh.finalize()













