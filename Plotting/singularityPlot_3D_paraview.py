import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import meshio
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plots data along a set of singular lines using data from specific points

metalHeights=["100nm", "1um", "2um", "4um", "6um", "8um", "10um"]
path = "../Results/"
fileList = [f"recCavity_Eplane_prisms_meshLength=1500e-6_p=1,1_s=0,0_volPts.npz", f"recCavity_Eplane_prisms_meshLength=1500e-6_p=1,1_s=1,1_volPts.npz", \
            f"recCavity_Eplane_prisms_meshLength=1500e-6_p=1,1_s=1,1_volPts_noTriEdgeFuncs.npz"]
            
origData = np.load(f"{path}{fileList[0]}", allow_pickle=True)
origFields = origData["pointFields"][0]

for fileInd, filename in enumerate(fileList):
  meshData = np.load(f"{path}{filename}", allow_pickle=True)
  points = meshData["fieldPoints"]
  fields = meshData["pointFields"][0]

  # Write data into the vtk file
  if fileInd == 0:
    # If this is the first file, add mesh info
    tetList = Delaunay(points[:])
    mesh = meshio.Mesh(
      points,
      cells={"tetra": tetList.simplices}
    )

  # Write the field values assocaited with each point
  # for ii in range(3):
  #   mesh.point_data[f"file{fileInd}_{ii}"] = np.real(fields[:,ii]) + np.imag(fields[:,ii])
  mesh.point_data[filename] = np.real(fields) + np.imag(fields)
  
  # Also write in errors
  mesh.point_data[f"err_{filename}"] = (np.real(fields) + np.imag(fields) - np.real(origFields) - np.imag(origFields)) \
                                        / np.linalg.norm(np.real(origFields) - np.imag(origFields), axis=1).reshape(len(origFields),1)

# Save the vtk
vtkFile = f"fields_{fileList[0][:-4]}.vtk"
meshio.write(vtkFile, mesh)