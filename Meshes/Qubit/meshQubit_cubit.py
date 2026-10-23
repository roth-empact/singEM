#!python
import numpy as np
import sys
_DEFAULT_CUBIT_PATH = "C:/Program Files/Coreform Cubit 2025.12/bin"
#_DEFAULT_CUBIT_PATH = "/depot/rothte/apps/Coreform-Cubit-2025.3/bin"
sys.path.append(f'{_DEFAULT_CUBIT_PATH}')
import cubit # type: ignore
cubit.init(['cubit', '-nojournal'])
cubit.cmd('reset')
sys.path.append("C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Meshes/Prisms")
# sys.path.append("/scratch/negishi/selkin/3D-FEM/Meshes/Prisms")

cubit.cmd("import acis \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Meshes/surfPart/IBM_surfPart.sat\" attributes_on separate_bodies")
# cubit.cmd("import acis \"/scratch/negishi/selkin/3D-FEM/Meshes/surfPart/IBM_surfPart.sat\" attributes_on separate_bodies")

############## Assemble some string definitions that will be used repeatedly
genBrick = "brick x {} y {} z {}"
genRecZ = "create surface rectangle width {} height {} zplane"
genCircZ = "create surface circle radius {} zplane"
genLine = "create curve location {} {} {} location {} {} {}"
subSurf = "subtract surface {} from surface {}"
subVol_keep = "subtract volume {} from volume {} keep_tool"
movVol = "move volume {} x {} y {} z {}"
cleanVol = "auto_clean volume {} small_curves small_curve_size {}"
sweepSurf = "sweep surface {} direction {} distance {} keep"
copyMesh = "copy mesh surface {} onto surface {} source curve {} source vertex {} target curve {} target vertex {}"
blockVol = "block {} add volume {}"
blockSurf = "block {} add surf {}"
blockLine = "block {} add curve {}"
nodesetVert = "nodeset {} add vertex {}"
nameBlock = "block {} name \"{}\""
nameNodeset = "nodeset {} name \"{}\""

############## Label geometry entities
bulkVol = 1 # Substrate volume
boxVol = 2 # Bounding box volume
gndVol = 4 # Ground plane volume
padVols = [3, 12] # PEC part of pads

padMAVols = [5, 6] # Pad metal-air interfaces
padMSVols = [8, 9]
saVol = 11 # Substrate-air interface
gndMAVol = 7 # Ground metal-air interface
gndMSVol = 10 # Ground metal-substrate interface
portVol = 13 # Cubit port volume

## Delete the interface volumes
for volSet in [padMAVols, padMSVols, [saVol, gndMAVol, gndMSVol]]:
  for vol in volSet:
    cubit.cmd(f"delete volume {vol}")

## Subtract the overlapping region of the port from the pads
for vol in padVols:
  cubit.cmd(subVol_keep.format(vol, portVol))
  portVol = cubit.get_last_id("volume")

## Create a volume in the gap region (for meshing)
botZPos = 0.0 # Height of the bottom metal surface
metalHeight = 1e-4
topZPos = metalHeight # Height of the top metal surface
boxLength, boxWidth = [0.95, 1.1] # Dimensions of the box
centerCoords = [0.785, -0.45, metalHeight/2]
cubit.cmd(genBrick.format(boxLength, boxWidth, metalHeight))
gapVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(gapVol, centerCoords[0], centerCoords[1], centerCoords[2]))
# Subtract the PEC volumes from the gap
for volSet in [padVols, [gndVol]]:
  for vol in volSet:
    cubit.cmd(subVol_keep.format(vol, gapVol))
    gapVol = cubit.get_last_id("volume")

# Create a seperate volume under the port (to avoid creating a trench under it)
portSurf = cubit.get_relatives("volume", portVol, "surface")[0]
cubit.cmd(f"sweep surface {portSurf} direction 0 0 -1 distance {metalHeight} keep")
portGapVol = cubit.get_last_id("volume")
cubit.cmd(subVol_keep.format(portGapVol, gapVol))

# Create the trench region below the gap, and create regions below 
trenchHeight = 1e-2
newVols = []
botSurfs = []
for vol in [gapVol, gndVol, padVols[0], padVols[1]]:
  gapSurfs = cubit.get_relatives("volume", vol, "surface")
  for surf in gapSurfs:
    if np.isclose(cubit.get_center_point("surface", surf)[2], botZPos, atol=metalHeight/10):
      botSurf = surf
      break
  cubit.cmd(f"sweep surface {botSurf} direction 0 0 -1 distance {trenchHeight} keep")
  newVols.append(cubit.get_last_id("volume"))
  if vol == gapVol:
    # Subtract regions between the pads out of the trench
    padCenters = [cubit.get_center_point("volume", vol) for vol in padVols]
    cubit.cmd(genBrick.format(np.abs(padCenters[0][0]-padCenters[1][0]), 20e-3, trenchHeight))
    tempVol = cubit.get_last_id("volume")
    portCenter = cubit.get_center_point("surface", portSurf)
    cubit.cmd(movVol.format(tempVol, portCenter[0], portCenter[1], -trenchHeight/2))
    cubit.cmd(f"subtract volume {tempVol} from volume {newVols[-1]}")
  cubit.cmd(subVol_keep.format(newVols[-1], bulkVol))
  botSurfs.append(botSurf)
trenchVol, gndSubVol = newVols[:2]
padSubVols = newVols[2:]


# Delete the port gap volume
# cubit.cmd(f"unite volume {gapVol} {portGapVol}")
cubit.cmd(f"unite volume {bulkVol} {portGapVol}")

# Subtract all other volumes from the top box to eliminate overlap
for vol in [gapVol, gndVol, padVols[0], padVols[1], bulkVol]:
  cubit.cmd(subVol_keep.format(vol, boxVol))
  # boxVol = cubit.get_last_id("volume")

## Add cutout boxes around the junction to avoid measuring participation ratios there
cutBoxSizes = [trenchHeight, trenchHeight, trenchHeight+metalHeight]
cubit.cmd(genBrick.format(cutBoxSizes[0], cutBoxSizes[1], cutBoxSizes[2]))
cutBoxVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(cutBoxVol, portCenter[0], portCenter[1], -(trenchHeight-metalHeight)/2))
# Subtract the cutout out of the substrate, and subtract the gap out of the cutout
cubit.cmd(subVol_keep.format(cutBoxVol, bulkVol))
cubit.cmd(subVol_keep.format(gapVol, cutBoxVol))

# Isolate the part of the pads overlapping with the cutout, then subtract the cutout
# from the substrate region below the pads
padCutVols = []
for ii in range(2):
  cubit.cmd(f"intersect volume {padVols[ii]} {cutBoxVol} keep")
  padCutVols.append(cubit.get_last_id("volume"))
  cubit.cmd(subVol_keep.format(padCutVols[ii], padVols[ii]))
  cubit.cmd(subVol_keep.format(padCutVols[ii], cutBoxVol))
  cubit.cmd(subVol_keep.format(padSubVols[ii], cutBoxVol))



cubit.cmd("imprint tolerant volume all tolerance 1e-06")
cubit.cmd("merge tolerance 1e-06")
cubit.cmd("merge volume all")

# Get the interface surfaces
bulkSurfs = cubit.get_relatives("volume", bulkVol, "surface")
boxSurfs = cubit.get_relatives("volume", boxVol, "surface")
gndSurfs = cubit.get_relatives("volume", gndVol, "surface")
padSurfs = np.array([cubit.get_relatives("volume", padVols[ii], "surface") for ii in range(2)]) # Combine surfaces for both pads
padSubSurfs = np.array([cubit.get_relatives("volume", padSubVols[ii], "surface") for ii in range(2)])
trenchSurfs = cubit.get_relatives("volume", trenchVol, "surface")
portSurf = cubit.get_relatives("volume", portVol, "surface")[0] # Only one surface for the port
gapSurfs = cubit.get_relatives("volume", gapVol, "surface")

# padMSSurfs = []
# for surf in bulkSurfs:
#   surfCoords = cubit.get_center_point("surface", surf)
#   if np.isclose(surfCoords[2], botZPos, atol=metalHeight/100):
#     # If the surface is along the top of the substrate, check which volume it also belongs to
#     if surf in padSurfs:
#       padMSSurfs.append(surf)
#     elif surf in gndSurfs:
#       gndMSSurf = surf
#   elif np.isclose(surfCoords[2], botZPos-trenchHeight, atol=metalHeight/100) and surf in trenchSurfs:
#     # The substrate height is lower in the gap region (where the trench is)
#     SASurf = surf

gndMSSurfs = [botSurfs[1]]
padMSSurfs = [[], []]
for ii in range(2):
  for surf in padSurfs[ii]:
    if surf in padSubSurfs[ii] or surf in bulkSurfs:
      padMSSurfs[ii].append(surf)

SASurfs = []
for surf in trenchSurfs:
  surfCoords = cubit.get_center_point("surface", surf)
  if not np.isclose(surfCoords[2], botZPos, atol=metalHeight/100):
    SASurfs.append(surf)
# Add the contribution of exposed substrate
for surf in bulkSurfs: 
  surfCoords = cubit.get_center_point("surface", surf)
  if np.isclose(surfCoords[2], botZPos, atol=metalHeight/100):
    SASurfs.append(surf)

padMASideSurfs = [[], []]
gndMASideSurfs = []
padMATopSurfs = [-1, -1]
for surfSet in [boxSurfs, gapSurfs]:
  for surf in surfSet:
    surfCoords = cubit.get_center_point("surface", surf)
    if surf in padSurfs[0]:
      if np.isclose(surfCoords[2], botZPos+metalHeight, atol=metalHeight/100):
        padMATopSurfs[0] = surf
      else:
        padMASideSurfs[0].append(surf)
    elif surf in padSurfs[1]:
      if np.isclose(surfCoords[2], botZPos+metalHeight, atol=metalHeight/100):
        padMATopSurfs[1] = surf
      else:
        padMASideSurfs[1].append(surf)
    elif surf in gndSurfs:
      if np.isclose(surfCoords[2], botZPos+metalHeight, atol=metalHeight/100):
        gndMATopSurf = surf
      else:
        gndMASideSurfs.append(surf)

## Identify all singular lines
# Start by isolating the x and y coordinates of the outer boundary
bulkSurfs = cubit.get_relatives("volume", bulkVol, "surface")
botSurf = bulkSurfs[0]
botSurfZ = cubit.get_center_point("surface", botSurf)[2]
for surf in bulkSurfs[1:]:
  surfCenter = cubit.get_center_point("surface", surf)
  if surfCenter[2] < botSurfZ:
    botSurf = surf
    botSurfZ = surfCenter[2]

botSurfLines = cubit.get_relatives("surface", botSurf, "curve")
lineCoords = cubit.get_center_point("curve", botSurfLines[0])
xLims = [lineCoords[0], lineCoords[0]]
yLims = [lineCoords[1], lineCoords[1]]
botZ = lineCoords[0]
for line in botSurfLines[1:]:
  lineCoords = cubit.get_center_point("curve", line)
  if lineCoords[0] < xLims[0]:
    xLims[0] = lineCoords[0]
  if lineCoords[0] > xLims[1]:
    xLims[1] = lineCoords[0]
  if lineCoords[1] < yLims[0]:
    yLims[0] = lineCoords[1]
  if lineCoords[1] > yLims[1]:
    yLims[1] = lineCoords[1]

boxSurfs = cubit.get_relatives("volume", boxVol, "surface")
topSurf = boxSurfs[0]
topSurfZ = cubit.get_center_point("surface", topSurf)[2]
for surf in boxSurfs[1:]:
  surfCenter = cubit.get_center_point("surface", surf)
  if surfCenter[2] < topSurfZ:
    topSurf = surf
    topSurfZ = surfCenter[2]
topZ = topSurfZ

zLims = [botZ, topZ]

# # Now pick out all singular lines
# singLines = []
# for vol in [gapVol]:
#   volSurfs = cubit.get_relatives("volume", vol, "surface")
#   for surf in volSurfs:
#     surfCenter = cubit.get_center_point("surface", surf)
#     if not ((np.isclose(surfCenter[0], xLims[0], atol=metalHeight/100)) or \
#             (np.isclose(surfCenter[0], xLims[1], atol=metalHeight/100)) or \
#             (np.isclose(surfCenter[1], yLims[0], atol=metalHeight/100)) or \
#             (np.isclose(surfCenter[1], yLims[1], atol=metalHeight/100)) or \
#             (np.isclose(surfCenter[2], zLims[0], atol=metalHeight/100)) or \
#             (np.isclose(surfCenter[2], zLims[1], atol=metalHeight/100))):
#       surfLines = cubit.get_relatives("surface", surf, "curve")
#       for line in surfLines:
#         if line not in singLines:
#           singLines.append(line)

singLines = []
pecLines = np.hstack([cubit.get_relatives("volume", vol, "curve") for vol in [padVols[0], padVols[1], padCutVols[0], padCutVols[1], gndVol]])
pecLines = np.unique(pecLines)
for vol in [gapVol]:
  volLines = cubit.get_relatives("volume", vol, "curve")
  for line in volLines:
    if (line in pecLines) and (line not in singLines):
      lineCenter = cubit.get_center_point("curve", line)
      if not ((np.isclose(lineCenter[0], xLims[0], atol=metalHeight/100)) or \
              (np.isclose(lineCenter[0], xLims[1], atol=metalHeight/100)) or \
              (np.isclose(lineCenter[1], yLims[0], atol=metalHeight/100)) or \
              (np.isclose(lineCenter[1], yLims[1], atol=metalHeight/100))):
        singLines.append(line)

# Check which of the lines are actually singular by drawing a cylinder around them and measuring how much is in PEC
realSingLines = []
cylRad = 1e-3
pecVols = [gndVol, padVols[0], padVols[1]]
for line in singLines:
  lineCenter = np.array(cubit.get_center_point("curve", line))
  lineVerts = cubit.get_relatives("curve", line, "vertex")
  lineVertPos = np.array([cubit.get_center_point("vertex", vert) for vert in lineVerts])
  lineDist = lineVertPos[1] - lineVertPos[0]
  cylHeight = np.linalg.norm(lineDist)/2
  lineDir = np.abs(lineDist) / np.linalg.norm(lineDist)

  cubit.cmd(genCircZ.format(cylRad))
  lineCircSurf = cubit.get_last_id("surface")
  cubit.cmd(f"sweep surface {lineCircSurf} direction z distance {cylHeight}")
  cylVol = cubit.get_last_id("volume")
  rotAxis = -np.abs(np.cross(np.array([0,0,1.0]), lineDir))
  if not np.all(np.isclose(rotAxis, np.zeros((3,), dtype=np.float64), atol=1e-3)):
    cubit.cmd(f"rotate volume {cylVol} angle 90 about origin 0 0 0 direction {rotAxis[0]} {rotAxis[1]} {rotAxis[2]}" )
  cubit.cmd(movVol.format(cylVol, *(lineVertPos[0] + (lineDist-cylHeight*lineDir)/2)))

  # Cut the cylinder out of the pec volumes
  for vol in pecVols:
    cubit.cmd(subVol_keep.format(vol, cylVol))

  # Check the cylinder volume to determine if line is singular (3pi/2 angle)
  cylVolume = cubit.volume(cylVol).volume()
  if cylVolume > (cylRad**2*np.pi/2)*cylHeight: # If the angle is greater than pi, treat the line as singular
    realSingLines.append(line)

  # Get rid of the volume
  cubit.cmd(f"delete volume {cylVol}")

singLines = np.array(realSingLines)

# Eliminate the lines that touch the port surface from the singular list
portVerts = cubit.get_relatives("volume", portVol, "vertex")
lineInd = 0
for line in singLines:
  lineVerts = np.array(cubit.get_relatives("curve", int(line), "vertex"))
  vertOnPort = np.array([lineVerts[ii] in portVerts for ii in range(2)]) # Flags whether each vertex is on the port
  if np.all(vertOnPort):
    # If the line is along the port, remove it from the singular list
    singLines = np.append(singLines[:lineInd], singLines[lineInd+1:])
  elif np.any(vertOnPort):
    # If the line is touching a vertex of the port, split it into a smaller segment neighboring the port
    singLines = np.append(singLines[:lineInd], singLines[lineInd+1:]) # First remove the existing line
    portVertCoords = cubit.get_center_point("vertex", int(lineVerts[vertOnPort][0]))
    oppVertCoords = cubit.get_center_point("vertex", int(lineVerts[~vertOnPort][0]))
    lineDir = np.array(oppVertCoords) - np.array(portVertCoords)
    lineLength = np.linalg.norm(lineDir)
    if lineLength > 1e-3:
      # Don't split the small vertical lines, just remove them altogether
      lineDir /= np.linalg.norm(lineDir)
      newVertCoords = portVertCoords + 2e-3*lineDir
      cubit.cmd(f"split curve {line} position {newVertCoords[0]} {newVertCoords[1]} {newVertCoords[2]}")
      cubit.cmd("imprint all")
      cubit.cmd("merge all")

      newSingLine = cubit.get_relatives("vertex", int(lineVerts[~vertOnPort][0]), "curve")[-1]
      singLines = np.append(singLines, newSingLine)
    
  else:
    lineInd += 1

############ Mesh the geometry
cubit.cmd("set trimesher surface gradation 2.0")
cubit.cmd("set trimesher volume gradation 2.0")

# Mesh the singular lines
# for line in singLines:
#   cubit.cmd(f"curve {line} size {0.01}")
#   cubit.cmd(f"mesh curve {line}")

## Mesh the port surface
portSurf = cubit.get_relatives("volume", portVol, "surface")[0]
cubit.cmd(f"surface {portSurf} size {0.001}")
cubit.cmd(f"surface {portSurf} scheme trimesh")
cubit.cmd(f"mesh surface {portSurf}")

## Mesh the gap and trench volume (and the volumes adjacent to the trench)
gapDensity = 0.03
for vol in [gapVol, trenchVol, gndSubVol, padSubVols[0], padSubVols[1], cutBoxVol]:
  # Mesh the top and bottom surfaces, then mesh the volume
  for surf in cubit.get_relatives("volume", vol, "surface"):
    if np.isclose(cubit.get_center_point("surface", surf)[2], botZ, rtol=metalHeight/10):
      cubit.cmd(f"surface {surf} size {gapDensity}")
      cubit.cmd(f"surface {surf} scheme trimesh")
      cubit.cmd(f"mesh surface {surf}")
      botSurf = surf
    elif vol != trenchVol and np.isclose(cubit.get_center_point("surface", surf)[2], topZ, rtol=metalHeight/10):
      # The top of the trench is the bottom of the gap, so it's already meshed
      cubit.cmd(f"surface {surf} size {gapDensity}")
      cubit.cmd(f"surface {surf} scheme trimesh")
      cubit.cmd(f"mesh surface {surf}")
      topSurf = surf
  cubit.cmd(f"volume {vol} size {gapDensity}")
  cubit.cmd(f"volume {vol} scheme tetmesh")
  cubit.cmd(f"mesh volume {vol}")

## Mesh the metal surfaces
gndDensity = 0.08
metalMeshSizes = [gapDensity, gapDensity, gapDensity, gapDensity, gndDensity]
for ii, vol in enumerate([padVols[0], padVols[1], padCutVols[0], padCutVols[1], gndVol]):
  # Start by meshing the bottom
  for surf in cubit.get_relatives("volume", vol, "surface"):
    if np.isclose(cubit.get_center_point("surface", surf)[2], botZ, rtol=metalHeight/10):
      cubit.cmd(f"surface {surf} size {metalMeshSizes[ii]}")
      cubit.cmd(f"surface {surf} scheme trimesh")
      cubit.cmd(f"mesh surface {surf}")
      botSurf = surf
    elif np.isclose(cubit.get_center_point("surface", surf)[2], topZ, rtol=metalHeight/10):
      cubit.cmd(f"surface {surf} size {metalMeshSizes[ii]}")
      cubit.cmd(f"surface {surf} scheme trimesh")
      cubit.cmd(f"mesh surface {surf}")
      topSurf = surf

## Mesh the outer surfaces of the GND plane
centerZ = botZ + metalHeight/2
gndOuterSurfs = []
for surf in cubit.get_relatives("volume", gndVol, "surface"):
  if np.isclose(cubit.get_center_point("surface", surf)[2], centerZ, rtol=metalHeight/10) \
    and cubit.get_surface_element_count(surf) == 0:
    cubit.cmd(f"surface {surf} size {metalMeshSizes[-1]}")
    cubit.cmd(f"surface {surf} scheme trimesh")
    cubit.cmd(f"mesh surface {surf}")
    gndOuterSurfs.append(surf)

## Mesh each volume
for vol in [bulkVol, boxVol]:
  cubit.cmd(f"volume {vol} size {0.2}")
  cubit.cmd(f"volume {vol} scheme tetmesh")
  cubit.cmd(f"mesh volume {vol}")

# ############# Assign blocks
blockCnt = 1 # Index 1 for the substrate
for vol in [bulkVol, gndSubVol, padSubVols[0], padSubVols[1], cutBoxVol]:
  cubit.cmd(blockVol.format(blockCnt, vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr10"))

blockCnt += 1 # Index 2 for the bounding box and gap volumes
for vol in [boxVol, gapVol, trenchVol]:
  cubit.cmd(blockVol.format(blockCnt, vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr1"))

blockCnt += 1 # Index 3 for the outer surface (and GND)
outerSurfs = [] # Holds the IDs for all the outer box surfaces
for surfList in [bulkSurfs, boxSurfs]:
  for surf in surfList:
    if (not np.isclose(cubit.get_center_point("surface", surf)[2], botZPos, atol=2*trenchHeight)):
      outerSurfs.append(surf)

# Also add outer surfaces for the gap volume under GND
gndSubSurfs = cubit.get_relatives("volume", gndSubVol, "surface")
for surf in gndSubSurfs:
  if np.isclose(cubit.get_center_point("surface", surf)[2], botZPos-trenchHeight/2, atol=metalHeight) \
    and surf not in trenchSurfs:
    outerSurfs.append(surf)

for surfList in [outerSurfs, gndOuterSurfs]:
  for surf in surfList:
    cubit.cmd(blockSurf.format(blockCnt, surf))
for surf in cubit.get_relatives("volume", gndVol, "surface"):
  if surf not in gndOuterSurfs:
    cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC0"))

blockCnt += 1 # Index 5 for the first pad surface surface
for vol in [padVols[0], padCutVols[0]]:
  for surf in cubit.get_relatives("volume", vol, "surface"):
    cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC1"))

blockCnt += 1 # Index 6 for the second pad surface surface
for vol in [padVols[1], padCutVols[1]]:
  for surf in cubit.get_relatives("volume", vol, "surface"):
    cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC2"))

blockCnt += 1 # Index 6 for the port surface
cubit.cmd(blockSurf.format(blockCnt, portSurf))
cubit.cmd(nameBlock.format(blockCnt, "Port1"))


cubit.cmd("set duplicate block elements on")
# Add metal air interfaces
surfSet = [gndMATopSurf, padMATopSurfs[0], padMATopSurfs[1]]
names = ["MATopGND", "MATopPad1", "MATopPad2"]
for ii, surf in enumerate(surfSet):
  blockCnt += 1 
  cubit.cmd(blockSurf.format(blockCnt, surf))
  cubit.cmd(nameBlock.format(blockCnt, names[ii])) 
  
# Add metal air side surfaces
names = ["MASideGND", "MASidePad1", "MASidePad2"]
for ii, surfSet in enumerate([gndMASideSurfs, padMASideSurfs[0], padMASideSurfs[1]]):
  blockCnt += 1 
  for surf in surfSet:
    cubit.cmd(blockSurf.format(blockCnt, surf))
  cubit.cmd(nameBlock.format(blockCnt, names[ii])) 

# Add metal substrate interfaces
surfSets = [gndMSSurfs, padMSSurfs[0], padMSSurfs[1]]
names = ["MSGND", "MSPad1", "MSPad2"]
for ii, surfSet in enumerate(surfSets):
  blockCnt += 1 
  for surf in surfSet:
    cubit.cmd(blockSurf.format(blockCnt, surf))
  cubit.cmd(nameBlock.format(blockCnt, names[ii])) 

# Add the substrate air interfaces
blockCnt += 1
for surf in SASurfs:
  cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "SA")) 

blockCnt += 1 # Add the singular edges
for curve in singLines:
  cubit.cmd(blockLine.format(blockCnt, curve))
cubit.cmd(nameBlock.format(blockCnt, "singLines"))

nodesetCnt = 1 # Create a nodeset for the singular vertices
singLineVerts = [] # Track all vertices connected to singular lines
for line in singLines:
  lineVerts = cubit.get_relatives("curve", int(line), "vertex")
  for vert in lineVerts:
    singLineVerts.append(vert) 

# Isolate the corners that are connected to three singular lines
singLineVerts = np.sort(np.array(singLineVerts))
singCorners = []
ii = 0
while ii < len(singLineVerts)-2:
  if singLineVerts[ii] == singLineVerts[ii+1]:
    if singLineVerts[ii] == singLineVerts[ii+2]:
      singCorners.append(singLineVerts[ii])
      ii += 3
    else:
      ii += 2
  else:
    ii += 1

for vert in singCorners:
  cubit.cmd(nodesetVert.format(nodesetCnt, vert))
cubit.cmd(nameNodeset.format(nodesetCnt, "singCorners"))


## Export the mesh
#cubit.cmd("export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Meshes/surfPart/IBMSurfPart_10umTrench_splitSurfs_coarsePort.inp\" mesh_only overwrite everything")
# cubit.cmd("export abaqus \"/scratch/negishi/selkin/3D-FEM/Meshes/surfPart/IBMSurfPart.inp\" mesh_only overwrite everything")












