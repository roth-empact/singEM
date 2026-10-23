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

############## Draw the geometry
## Set sizes
subLength = 800
subHeight = 45
topBoxHeight = 400
botBoxHeight = topBoxHeight - subHeight

traceWidth = 10
gapWidth = 25

portLength = 10

centralLength = 100

## Draw the metal surface and cut out two linear traces
cubit.cmd(genRecZ.format(subLength, subLength))
gndBody = cubit.get_last_id("body")

for ii in range(2):
  cubit.cmd(genRecZ.format(gapWidth, subLength))
  vol = cubit.get_last_id("volume")
  cubit.cmd(movVol.format(vol, (2*ii-1)*(gapWidth+traceWidth)/2, 0, 0))
  cubit.cmd(f"subtract volume {vol} from body {gndBody}")

## Isolate each of the conductor volumes
condVols = cubit.get_relatives("body", gndBody, "volume")
for vol in condVols:
  volCenter = cubit.get_center_point("volume", vol)
  if volCenter[0] < -(traceWidth+gapWidth)/2:
    leftGNDVol = vol
  elif volCenter[0] > (traceWidth+gapWidth)/2:
    rightGNDVol = vol
  else:
    traceVol = vol
cubit.cmd(f"split body {gndBody}")

# Add ports to the end of the traces on each side
portVols = []
for ii in range(2):
  cubit.cmd(genRecZ.format(traceWidth, portLength))
  portVols.append(cubit.get_last_id("volume"))
  cubit.cmd(movVol.format(portVols[ii], 0, (2*ii-1)*(subLength-portLength)/2, 0))
  cubit.cmd(subVol_keep.format(portVols[ii], traceVol))

# Create surfaces to mark the central region on each conductor
cubit.cmd(genRecZ.format(subLength, centralLength))
msBody = cubit.get_last_id("body")

# Subtract the body from the substrate surface
for ii in range(2):
  cubit.cmd(genRecZ.format(gapWidth, subLength))
  vol = cubit.get_last_id("volume")
  cubit.cmd(movVol.format(vol, (2*ii-1)*(gapWidth+traceWidth)/2, 0, 0))
  cubit.cmd(f"subtract volume {vol} from body {msBody}")

# Isolate each surface
msVols = cubit.get_relatives("body", msBody, "volume")
for vol in msVols:
  volCenter = cubit.get_center_point("volume", vol)
  if volCenter[0] < -(traceWidth+gapWidth)/2:
    leftGNDMSVol = vol
  elif volCenter[0] > (traceWidth+gapWidth)/2:
    rightGNDMSVol = vol
  else:
    traceMSVol = vol
cubit.cmd(f"split body {msBody}")

# Draw the substrate and bounding boxes
cubit.cmd(genBrick.format(subLength, subLength, subHeight))
bulkVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(bulkVol, 0, 0, -subHeight/2))
cubit.cmd(genBrick.format(subLength, subLength, topBoxHeight))
topBoxVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(topBoxVol, 0, 0, topBoxHeight/2))
# cubit.cmd(genBrick.format(subLength, subLength, botBoxHeight))
# botBoxVol = cubit.get_last_id("volume")
# cubit.cmd(movVol.format(botBoxVol, 0, 0, -subHeight-botBoxHeight/2))

# Create additional boxes for measuring the total field in the relevant region
bulkBody = cubit.get_relatives("volume", bulkVol, "body")[0]
cubit.cmd(genBrick.format(subLength, centralLength, subHeight))
bulkMeasVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(bulkMeasVol, 0, 0, -subHeight/2))
cubit.cmd(subVol_keep.format(bulkMeasVol, bulkVol))
bulkVols = list(cubit.get_relatives("body", bulkBody, "volume")) # Get the IDs of the split volumes
bulkVols.append(bulkMeasVol)
cubit.cmd(f"split body {bulkBody}")

boxBody = cubit.get_relatives("volume", topBoxVol, "body")[0]
cubit.cmd(genBrick.format(subLength, centralLength, topBoxHeight))
boxMeasVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(boxMeasVol, 0, 0, topBoxHeight/2))
cubit.cmd(subVol_keep.format(boxMeasVol, topBoxVol))
boxVols = list(cubit.get_relatives("body", boxBody, "volume")) # Get the IDs of the split volumes
boxVols.append(boxMeasVol)
cubit.cmd(f"split body {boxBody}")

cubit.cmd("imprint all")
cubit.cmd("merge all")

# Record all of the ground and signal surfaces
gndSurfs, sigSurfs = [[], []]
portSurfs = np.array([cubit.get_relatives("volume", vol, "surface")[0] for vol in portVols])
bulkSurfs = []
for vol in bulkVols:
  bulkSurfs.extend(cubit.get_relatives("volume", vol, "surface"))
bulkSurfs = np.unique(np.array(bulkSurfs))

for surf in bulkSurfs:
  surfCenter = cubit.get_center_point("surface", int(surf))
  if np.isclose(surfCenter[0], -subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[0], subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[1], -subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[1], subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[2], -subHeight, atol=portLength/10):
    # If the surface is near the outer boundary, it's PEC
    gndSurfs.append(surf)
  elif np.isclose(surfCenter[2], 0.0, atol=subHeight/10):
    if np.isclose(surfCenter[0], 0.0, atol=traceWidth/2) and surf not in portSurfs:
      # If it's in the center (and not a port), it's part of the signal trace
      sigSurfs.append(surf)
    elif surfCenter[0] < -gapWidth-traceWidth or surfCenter[0] > gapWidth+traceWidth:
      gndSurfs.append(surf) # Top ground surfaces

# Also record ground surfaces in the top box
boxSurfs = []
for vol in boxVols:
  boxSurfs.extend(cubit.get_relatives("volume", vol, "surface"))
boxSurfs = np.unique(np.array(boxSurfs))
for surf in boxSurfs:
  surfCenter = cubit.get_center_point("surface", int(surf))
  if np.isclose(surfCenter[0], -subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[0], subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[1], -subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[1], subLength/2, atol=portLength/10) \
  or np.isclose(surfCenter[2], topBoxHeight, atol=portLength/10):
    # If the surface is on the outer boundary, it's PEC
    gndSurfs.append(surf)

## Singular lines
# Start by isolating the gap surfaces
gapSurfs = []
for surf in bulkSurfs:
  surfCenter = cubit.get_center_point("surface", int(surf))
  if np.isclose(surfCenter[0], -(gapWidth+traceWidth)/2, atol=traceWidth/2) \
  or np.isclose(surfCenter[0], (gapWidth+traceWidth)/2, atol=traceWidth/2):
    gapSurfs.append(surf)

# Now pick out the singular lines
singLines = []
for surf in gapSurfs:
  for line in cubit.get_relatives("surface", int(surf), "curve"):
    lineCenter = cubit.get_center_point("curve", line)
    if not np.isclose(lineCenter[1], -subLength/2, atol=traceWidth/2) \
    and not np.isclose(lineCenter[1], subLength/2, atol=traceWidth/2) \
    and not np.isclose(lineCenter[1], -centralLength/2, atol=traceWidth/2) \
    and not np.isclose(lineCenter[1], centralLength/2, atol=traceWidth/2):
      singLines.append(line)

# Eliminate the lines that touch the port surface from the singular list
for vol in portVols:
  portVerts = cubit.get_relatives("volume", vol, "vertex")
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
      if lineLength > 1:
        # Don't split the small vertical lines, just remove them altogether
        lineDir /= np.linalg.norm(lineDir)
        newVertCoords = portVertCoords + 1*lineDir
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

## Mesh the ports fine
for surf in portSurfs:
  cubit.cmd(f"surface {surf} size {4}")
  cubit.cmd(f"surface {surf} scheme trimesh")
  cubit.cmd(f"mesh surface {surf}")

## Mesh the integration regions
traceSize = 20
sizes = [traceSize, 30, 30]
traceMSSurf = cubit.get_relatives("volume", traceMSVol, "surface")[0]
leftGNDMSSurf = cubit.get_relatives("volume", leftGNDMSVol, "surface")[0]
rightGNDMSSurf = cubit.get_relatives("volume", rightGNDMSVol, "surface")[0]
for ii, surf in enumerate([traceMSSurf, leftGNDMSSurf, rightGNDMSSurf]):
  cubit.cmd(f"surface {surf} size {sizes[ii]}")
  cubit.cmd(f"surface {surf} scheme trimesh")
  cubit.cmd(f"mesh surface {surf}")

## Mesh the rest of the trace
for surf in sigSurfs:
  cubit.cmd(f"surface {surf} size {traceSize}")
  cubit.cmd(f"surface {surf} scheme trimesh")
  cubit.cmd(f"mesh surface {surf}")

## Mesh the volumes
cubit.cmd(f"volume {bulkMeasVol} size {35}")
cubit.cmd(f"volume {bulkMeasVol} scheme tetmesh")
cubit.cmd(f"mesh volume {bulkMeasVol}")

cubit.cmd(f"volume {boxMeasVol} size {65}")
cubit.cmd(f"volume {boxMeasVol} scheme tetmesh")
cubit.cmd(f"mesh volume {boxMeasVol}")

sizes = [50, 100]
for ii, volList in enumerate([bulkVols, boxVols]):
  for vol in volList:
    if vol is not bulkMeasVol and vol is not boxMeasVol:
      cubit.cmd(f"volume {vol} size {sizes[ii]}")
      cubit.cmd(f"volume {vol} scheme tetmesh")
      cubit.cmd(f"mesh volume {vol}")

# ############# Assign blocks
blockCnt = 1 # Index 1 for the substrate (measurement volume)
cubit.cmd(blockVol.format(blockCnt, bulkMeasVol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr11p9_meas"))

blockCnt += 1 # Index 2 for the bounding box volume (measurement volume)
cubit.cmd(blockVol.format(blockCnt, boxMeasVol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr1_meas"))

blockCnt += 1 # Index 3 for the remaining substrate volumes
for vol in bulkVols: 
  if vol != bulkMeasVol:
    cubit.cmd(blockVol.format(blockCnt, vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr11p9"))

blockCnt += 1 # Index 4 for the remaining bounding box volume
for vol in boxVols:
  if vol != boxMeasVol:
    cubit.cmd(blockVol.format(blockCnt, vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr1"))

blockCnt += 1 # Index 5 for the outer surface (and GND)
for surf in gndSurfs:
  cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC0"))

blockCnt += 1 # Index 6 for the signal trace
for surf in sigSurfs:
  cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC1"))

for ii, surf in enumerate(portSurfs):
  blockCnt += 1 # Indexes 7 and 8 for the port surfaces
  cubit.cmd(blockSurf.format(blockCnt, surf))
  cubit.cmd(nameBlock.format(blockCnt, f"Port{ii}"))

# MS Interfaces
cubit.cmd("set duplicate block elements on")
for ii, surf in enumerate([traceMSSurf, leftGNDMSSurf, rightGNDMSSurf]):
  blockCnt += 1 
  cubit.cmd(blockSurf.format(blockCnt, surf))
  cubit.cmd(nameBlock.format(blockCnt, f"MSint_{ii}")) 

# Add singular lines to a block
blockCnt += 1
for line in singLines:
  cubit.cmd(blockLine.format(blockCnt, line))
cubit.cmd(nameBlock.format(blockCnt, f"singLines")) 

## Export the mesh
# cubit.cmd("export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Meshes/surfPart/IBMSurfPart_fineEdgeRegion.inp\" mesh_only overwrite everything")
# cubit.cmd("export abaqus \"/scratch/negishi/selkin/3D-FEM/Meshes/surfPart/IBMSurfPart_fineEdgeRegion.inp\" mesh_only overwrite everything")













