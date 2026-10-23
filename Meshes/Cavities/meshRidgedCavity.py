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
subVol = "subtract volume {} from volume {}"
subVol_keep = "subtract volume {} from volume {} keep_tool"
uniteVol = "unite volume {} {}"
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
# Set sizing parameters
cavLength = 0.01
cavWidth = 0.005
cavHeight = 0.0075
discHeight = cavHeight / 10

# Create the cavity itself
cubit.cmd(genBrick.format(cavWidth, cavLength, cavHeight))
vol = cubit.get_last_id("volume")

# Create the rectangular discontinuity
cubit.cmd(genBrick.format(cavWidth, cavLength/2, discHeight))
discVol = cubit.get_last_id("volume")
cubit.cmd(movVol.format(discVol, 0, -cavLength/4, 0))

# Subtract the discontinuity from the cavity
cubit.cmd(subVol.format(discVol, vol))
vol = cubit.get_last_id("volume")

# # Imprint and merge
# cubit.cmd("imprint all")
# cubit.cmd("merge all")

# Mark the singular lines as such
singLineCenters = np.array([[0, 0, -discHeight/2], [0, 0, discHeight/2]], dtype=np.float64)
lines = cubit.get_relatives("volume", vol, "curve")
singLines = []
for line in lines:
  lineCenter = np.array(cubit.get_center_point("curve", line))
  for singLineCenter in singLineCenters:
    if np.all(np.isclose(lineCenter, singLineCenter, atol=1e-4)):
      singLines.append(line)

############ Mesh the geometry
# ## Mesh the discontinuity surface
# discSurf = cubit.get_relatives("volume", discVol, "surface")[0]
# cubit.cmd(f"surface {discSurf} size {2e-3}") # 2e-3 for coarse mesh,4.1e-4 for fine
# cubit.cmd(f"surface {discSurf} scheme trimesh")
# cubit.cmd(f"mesh surface {discSurf}")

cubit.cmd("set trimesher surface gradation 2")
cubit.cmd("set trimesher volume gradation 2")

## Mesh the volume
meshLength = 0.4e-3
cubit.cmd(f"volume {vol} size {meshLength}") # 2e-3 for coarse mesh, 4.1e-4 for fine
cubit.cmd(f"volume {vol} tetmesh growth_factor {2.0}")
cubit.cmd(f"volume {vol} scheme tetmesh")
cubit.cmd(f"mesh volume {vol}")

# ############# Assign blocks
blockCnt = 1 # Index 1 for the volume
cubit.cmd(blockVol.format(blockCnt, vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr1"))

blockCnt += 1 # Outer surfaces
surfs = np.array(cubit.get_relatives("volume", vol, "surface"))
for surf in surfs:
  cubit.cmd(blockSurf.format(blockCnt, surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC0"))

blockCnt += 1 # Singular edges
for curve in singLines:
  cubit.cmd(blockLine.format(blockCnt, curve))
cubit.cmd(nameBlock.format(blockCnt, "singLines"))

# Export the mesh
cubit.cmd(f"export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Meshes/Cavities/ridgedCavity_meshLength={int(round(meshLength*1e6))}e-6.inp\" mesh_only overwrite everything")









