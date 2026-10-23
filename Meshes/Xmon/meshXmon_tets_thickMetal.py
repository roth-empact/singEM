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

cubit.cmd("import acis \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Meshes/Prisms/soloXmon/Ma-xmon_only.sat\" attributes_on")
# cubit.cmd("import acis \"/scratch/negishi/selkin/3D-FEM/Meshes/Prisms/Ma-xmon_only.sat\" attributes_on")

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
bulkVol = 5 # Substrate volume
boxVol = 6 # Bounding box volume
# Subtract the substrate from the box
cubit.cmd(subVol_keep.format(bulkVol, boxVol))
botBoxVol = cubit.get_last_id("volume")
topBoxVol = botBoxVol - 1
boxBody = 6
cubit.cmd(f"split body {boxBody}")
cubit.cmd(f"delete volume {botBoxVol}")
bulkSurfs = cubit.get_relatives("volume", bulkVol, "surface")
topBoxSurfs = cubit.get_relatives("volume", topBoxVol, "surface")

############# Grow the surface layers to volumes
# cubit.cmd("imprint all")
# cubit.cmd("merge all")

## Extend each of the surface layers
metalHeight = 300e-4 # Height of the metal layer
scaleFactor = 1e2 # Add scaling to aid with cleanup later
transSurf = 4 # Surface of the transmon metal
gapSurf = 6 # Gap between the metal layers
gndSurf = 1 # GND surface
portSurf = 2 # Port surface (for the qubit)
portSurf2 = 3
volList = []
for surf in [transSurf, gndSurf, portSurf, portSurf2, gapSurf]:
  cubit.cmd(sweepSurf.format(surf, "z", metalHeight))
  volList.append(cubit.get_last_id("volume"))
[transVol, gndVol, portVol, portVol2, gapVol] = volList
botZ = cubit.get_center_point("surface", transSurf)[2]
topZ = botZ + metalHeight

## Subtract all of the metal layers from the gap volume
for vol in [gndVol, portVol, portVol2, transVol]:
  cubit.cmd(subVol_keep.format(vol, gapVol))
  gapVol = cubit.get_last_id("volume")

## Isolate the SQUID and gap volumes
squidVol = gapVol
gapVol = squidVol - 1

## Subtract the new volumes from the box
squidBody = 13
cubit.cmd(f"split body {squidBody}") # Split up the squid and the rest of the gap
for vol in [transVol, gapVol, gndVol, portVol, portVol2, squidVol]:
  cubit.cmd(subVol_keep.format(vol, topBoxVol))

## Scale the bulk volume to be shorter
cubit.cmd(f"volume {bulkVol} scale x 1 y 1 z {0.25} about 0 0 0")

cubit.cmd("imprint tolerant volume all tolerance 1e-06")
cubit.cmd("merge tolerance 1e-06")
cubit.cmd("merge volume all")

# # Need to repeat to resolve issues with some surfaces
# cubit.cmd("imprint tolerant volume all tolerance 1e-06")
# cubit.cmd("merge tolerance 1e-06")
# cubit.cmd("merge volume all")

## Delete the surface bodies to aid with squaring off the volumes
for ii in range(4):
  cubit.cmd(f"delete body {ii+1}")

## Square off the sides of the volumes
transCenter = cubit.get_center_point("volume", transVol)
boxDims = np.array([0.6, 0.6, 1], dtype=np.float64)
cutoffPts = np.array([[-boxDims[0]/2, 0, 0], [0, -boxDims[1]/2, 0], \
                                    [boxDims[0]/2, 0, 0], [0, boxDims[1]/2, 0]], dtype=np.float64)
for ii in range(4):
  # Subtract away the sides
  cubit.cmd(genBrick.format(5*boxDims[0], 5*boxDims[1], 5*boxDims[2]))
  tempVol = cubit.get_last_id("volume")
  newCenter = 6*cutoffPts[ii] + transCenter
  cubit.cmd(movVol.format(tempVol, newCenter[0], newCenter[1], newCenter[2]))
  for vol in [topBoxVol, bulkVol, gndVol]:
    cubit.cmd(subVol_keep.format(tempVol, vol))
  cubit.cmd(f"delete volume {tempVol}")

cubit.cmd("imprint tolerant volume all tolerance 1e-06")
cubit.cmd("merge tolerance 1e-06")
cubit.cmd("merge volume all")

# # Need to repeat to resolve issues with some surfaces
# cubit.cmd("imprint tolerant volume all tolerance 1e-06")
# cubit.cmd("merge tolerance 1e-06")
# cubit.cmd("merge volume all")

## Add cylinders for mesh numbering
targetLines = [98, 101]
targetLineVerts = [cubit.get_relatives("curve", line, "vertex") for line in targetLines]
pecVols = [gndVol, transVol]
airVols = [gapVol, topBoxVol]
cylRads = [0.011, 0.015] #1/128
cylVols = []
for ind, lineVerts in enumerate(targetLineVerts):
  # Create a cylinder along the line for meshing purposes
  lineVertPos = np.array([cubit.get_center_point("vertex", vert) for vert in lineVerts])
  lineDist = lineVertPos[1] - lineVertPos[0]
  lineDir = lineDist / np.linalg.norm(lineDist)

  cubit.cmd(genCircZ.format(cylRads[ind]))
  lineCircSurf = cubit.get_last_id("surface")
  cylHeight = 1.0*cylRads[ind]
  cubit.cmd(f"sweep surface {lineCircSurf} direction z distance {cylHeight}")
  cylVol = cubit.get_last_id("volume")
  rotAxis = -np.abs(np.cross(np.array([0,0,1.0]), lineDir))
  if not np.all(np.isclose(rotAxis, np.zeros((3,), dtype=np.float64), atol=1e-3)):
    cubit.cmd(f"rotate volume {cylVol} angle 90 about origin 0 0 0 direction {rotAxis[0]} {rotAxis[1]} {rotAxis[2]}" )
  cubit.cmd(movVol.format(cylVol, *(lineVertPos[0] + (lineDist-cylHeight*lineDir)/2)))

  # Cut the cylinder out of the pec volumes
  for vol in pecVols:
    cubit.cmd(subVol_keep.format(vol, cylVol))

  # At this point, could check the cylinder volume to determine if line is singular (3pi/2 angle)

  # Subtract the other cylinders from this cylinder
  for vol in cylVols:
    cubit.cmd(subVol_keep.format(vol, cylVol))

  # Subtract the air out of the cylinders
  for vol in airVols:
    cubit.cmd(subVol_keep.format(cylVol, vol))

  cylVols.append(cylVol)

cubit.cmd("imprint tolerant volume all tolerance 1e-06")
cubit.cmd("merge tolerance 1e-06")
cubit.cmd("merge volume all")

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
zLims = [botZ, topZ]

# Now pick out all singular lines
singLines = []
for vol in [transVol, gndVol]:
  volSurfs = cubit.get_relatives("volume", vol, "surface")
  for surf in volSurfs:
    surfCenter = cubit.get_center_point("surface", surf)
    if not ((np.isclose(surfCenter[0], xLims[0], atol=metalHeight/100)) or \
            (np.isclose(surfCenter[0], xLims[1], atol=metalHeight/100)) or \
            (np.isclose(surfCenter[1], yLims[0], atol=metalHeight/100)) or \
            (np.isclose(surfCenter[1], yLims[1], atol=metalHeight/100)) or \
            (np.isclose(surfCenter[2], zLims[0], atol=metalHeight/100)) or \
            (np.isclose(surfCenter[2], zLims[1], atol=metalHeight/100))):
      surfLines = cubit.get_relatives("surface", surf, "curve")
      for line in surfLines:
        if line not in singLines:
          singLines.append(line)

# Check which of the lines are actually singular by drawing a cylinder around them and measuring how much is in PEC
realSingLines = []
cylRad = 1e-3
pecVols = [gndVol, transVol]
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

singLines = realSingLines

# # Add the modified singular lines back into the lists
# for ii, vol in enumerate(cylVols):
#   volLines = cubit.get_relatives("volume", vol, "curve")
#   for line in volLines:
#     lineVerts = cubit.get_relatives("curve", line, "vertex")
#     if lineVerts[0] == singLineVerts[ii][0] or lineVerts[0] == singLineVerts[ii][1] \
#        or lineVerts[1] == singLineVerts[ii][0] or lineVerts[1] == singLineVerts[ii][1]:
#       vertCenters = np.array([cubit.get_center_point("vertex", vert) for vert in lineVerts])
#       lineDist = vertCenters[1] - vertCenters[0]
#       lineDir = lineDist / np.linalg.norm()
#       if np.all(np.isclose(lineDir, lineDirs[ii], atol=1e-6)):
#         if line not in singLines:
#           singLines.append(line)

############ Mesh the geometry
cubit.cmd("set trimesher surface gradation 1.5")
cubit.cmd("set trimesher volume gradation 1.5")

# Mesh the singular lines
# for line in singLines:
#   cubit.cmd(f"curve {line} size {0.01}")
#   cubit.cmd(f"mesh curve {line}")

## Mesh the port volumes
for vol in [portVol, portVol2]:
  cubit.cmd(f"volume {vol} size {0.001}")
  cubit.cmd(f"volume {vol} scheme tetmesh")
  cubit.cmd(f"mesh volume {vol}")

for vol in cylVols:
  #cubit.cmd(f"volume {vol} size {0.1}")
  cubit.cmd(f"volume {vol} interval {1}")
  cubit.cmd(f"volume {vol} scheme tetmesh")
  cubit.cmd(f"mesh volume {vol}")

## Mesh the gap (and SQUID) volume
gapMeshSizes = [0.005, 0.012]
for ii, vol in enumerate([squidVol, gapVol]):
  cubit.cmd(f"volume {vol} size {gapMeshSizes[ii]}")
  cubit.cmd(f"volume {vol} scheme tetmesh")
  cubit.cmd(f"mesh volume {vol}")

## Mesh the metal surfaces
metalMeshSizes = [0.01, 0.05]
for ii, vol in enumerate([transVol, gndVol]):
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
  # # Then copy onto the top
  # botCurve = cubit.get_relatives("surface", botSurf, "curve")[0]
  # botVert = cubit.get_relatives("curve", botCurve, "vertex")[0]
  # botCurveCenter = cubit.get_center_point("curve", botCurve)
  # botVertCenter = cubit.get_center_point("vertex", botVert)
  # for curve in cubit.get_relatives("surface", topSurf, "curve"):
  #   curveCenter = cubit.get_center_point("curve", curve)
  #   if np.logical_and(np.isclose(curveCenter[0], botCurveCenter[0], rtol=metalHeight/10), \
  #                     np.isclose(curveCenter[0], botCurveCenter[0], rtol=metalHeight/10)):
  #     topCurve = curve
  #     for vert in cubit.get_relatives("curve", curve, "vertex"):
  #       vertCenter = cubit.get_center_point("vertex", botVert)
  #       if np.logical_and(np.isclose(vertCenter[0], botVertCenter[0], rtol=metalHeight/10), \
  #                     np.isclose(vertCenter[0], botVertCenter[0], rtol=metalHeight/10)):
  #         topVert = vert
  #     break
  # cubit.cmd(copyMesh.format(botSurf, topSurf, botCurve, botVert, topCurve, topVert))

## Mesh the outer surfaces of the GND plane
centerZ = botZ + metalHeight/2
gndOuterSurfs = []
for surf in cubit.get_relatives("volume", gndVol, "surface"):
  if np.isclose(cubit.get_center_point("surface", surf)[2], centerZ, rtol=metalHeight/10) \
    and cubit.get_surface_element_count(surf) == 0:
    cubit.cmd(f"surface {surf} size {metalMeshSizes[1]}")
    cubit.cmd(f"surface {surf} scheme trimesh")
    cubit.cmd(f"mesh surface {surf}")
    gndOuterSurfs.append(surf)

## Mesh each volume
for vol in [bulkVol, topBoxVol]:
  cubit.cmd(f"volume {vol} size {0.1}")
  cubit.cmd(f"volume {vol} scheme tetmesh")
  cubit.cmd(f"mesh volume {vol}")

############# Assign blocks
transZ = cubit.get_center_point("surface", transSurf)[2]
botSubSurf = 7 # Index for the bottom substrate surface, needs to be removed to ensure it isn't treated as PEC
outerSurfs = [] # Holds the IDs for all the outer box surfaces
topBoxSurfs = cubit.get_relatives("volume", topBoxVol, "surface")
for surfList in [bulkSurfs, topBoxSurfs]:
  for surf in surfList:
    if (not np.isclose(cubit.get_center_point("surface", surf)[2], transZ, atol=2*metalHeight)) \
      and (surf is not botSubSurf):
      outerSurfs.append(surf)

blockCnt = 1 # Index 1 for the substrate
cubit.cmd(blockVol.format(blockCnt, bulkVol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr10"))

blockCnt += 1 # Index 2 for the bounding box and gap volumes
for vol in [portVol, portVol2, topBoxVol, gapVol, squidVol]:
  cubit.cmd(blockVol.format(blockCnt, vol))
for vol in cylVols:
  cubit.cmd(blockVol.format(blockCnt, vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr1"))

blockCnt += 1 # Index 3 for the outer surface (and GND)
for surfList in [outerSurfs, gndOuterSurfs]:
  for surf in surfList:
    cubit.cmd(blockSurf.format(blockCnt, surf))
for surf in cubit.get_relatives("volume", gndVol, "surface"):
  if surf not in gndOuterSurfs:
    cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC0"))

blockCnt += 1 # Index 5 for the xmon surface
for surf in cubit.get_relatives("volume", transVol, "surface"):
  cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC1"))

blockCnt += 1 # Index 6 for the port surface
for vol in [portVol, portVol2]:
  for surf in cubit.get_relatives("volume", vol, "surface"):
    if np.isclose(cubit.get_center_point("surface", surf)[2], topZ, rtol=metalHeight/10):
      cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "Port1"))

blockCnt += 1 # Index 7 for the xmon metal-air interfaces
cubit.cmd("set duplicate block elements on")
for surf in cubit.get_relatives("volume", transVol, "surface"):
  surfCenter = cubit.get_center_point("surface", surf)
  if not np.isclose(surfCenter[2], botZ, atol=metalHeight/100): # Make sure it's not the surface connected to the substrate
    cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "MAint")) 

blockCnt += 1 # Index 8 for the singular edges
for curve in singLines:
  cubit.cmd(blockLine.format(blockCnt, curve))
cubit.cmd(nameBlock.format(blockCnt, "singLines"))

nodesetCnt = 1 # Create a nodeset for the singular vertices
singLineVerts = [] # Track all vertices connected to singular lines
for line in singLines:
  lineVerts = cubit.get_relatives("curve", line, "vertex")
  for vert in lineVerts:
    singLineVerts.append(vert) 

# Isolate the corners that are connected to three singular lines
singLineVerts = np.sort(np.array(singLineVerts))
singCorners = []
ii = 0
while ii < len(singLineVerts)-1:
  if singLineVerts[ii] == singLineVerts [ii+1]:
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
# cubit.cmd("export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code//CQEM-FEM/3D_Potential_FEM/Meshes/Prisms/xmon_tets_thickMetal_singLines.inp\" mesh_only overwrite everything")








