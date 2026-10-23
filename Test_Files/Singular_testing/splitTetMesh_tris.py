#!python
import numpy as np
import sys
_DEFAULT_CUBIT_PATH = "C:/Program Files/Coreform Cubit 2025.12/bin"
#_DEFAULT_CUBIT_PATH = "/depot/rothte/apps/Coreform-Cubit-2025.3/bin"
sys.path.append(f'{_DEFAULT_CUBIT_PATH}')
import cubit # type: ignore
cubit.init(['cubit', '-nojournal'])
cubit.cmd('reset')

# Create a unit tet
coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
for ii in range(len(coords)):
  cubit.cmd(f"create vertex location {coords[ii][0]} {coords[ii][1]} {coords[ii][2]}")
faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32) + 1
for face in faceNums:
  cubit.cmd(f"create surface vertex {face[0]} {face[1]} {face[2]}")
cubit.cmd(f"create volume surface 1 2 3 4 heal")
tetVol = cubit.get_last_id("volume")

# Mesh it
cubit.cmd(f"volume {tetVol} size 10")
cubit.cmd(f"volume {tetVol} scheme tetmesh")
cubit.cmd(f"mesh volume {tetVol}")

# Split it into a set of smaller tets
numsplit = 4
cubit.cmd(f"refine tet 1 NumSplit {numsplit}")

# Add the triangles and tet to blocks
blockCnt = 1
cubit.cmd(f"block {blockCnt} add volume {tetVol}")
for ii in range(4):
  blockCnt += 1
  cubit.cmd(f"block {blockCnt} add surface {ii+1}")

# Export the mesh
# cubit.cmd(f"export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Test_Files/Singular_testing/tetSplit_numSplit={numsplit}_tris.inp\" mesh_only overwrite everything")
# cubit.cmd("export abaqus \"/scratch/negishi/selkin/3D-FEM/Meshes/tetSplit_numSplit={numsplit}_tris.inp\" mesh_only overwrite everything")










