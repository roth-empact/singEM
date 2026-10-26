import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import meshio
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plots data along a 2D plane using data from specific points
path = "./Results/"
fileList = [f"qubitSurfPart.npz"]
            
origData = np.load(f"{path}{fileList[0]}", allow_pickle=True)
origFields = origData["pointFields"][0]

for fileInd, filename in enumerate(fileList):
  meshData = np.load(f"{path}{filename}", allow_pickle=True)
  points = 1e3*meshData["fieldPoints"]
  fields = meshData["pointFields"][0]

  # Identify the tangential directions of the triangle
  tanDirs = np.zeros((3,), dtype=np.bool_)
  for ii in range(3):
    if points[0,ii] != points[-1,ii]:
      # If the first and final point in this direction are not the same, it is tangential to the rectangle
      tanDirs[ii] = True

  # Write data into the vtk file
  if fileInd == 0:
    # If this is the first file, add mesh info
    triList = Delaunay(points[:,tanDirs])
    mesh = meshio.Mesh(
      points,
      cells={"triangle": triList.simplices}
    )

  # Write the field values assocaited with each point
  mesh.point_data[filename] = np.real(fields) + np.imag(fields)

# Save the vtk
vtkFile = f"{path}fields_{fileList[0][:-4]}.vtk"
meshio.write(vtkFile, mesh)