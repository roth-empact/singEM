import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import meshio
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plots data along a set of singular lines using data from specific points

# This verison writes a VTK file that can be opened in Paraview
# path = "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/"
# fileList = ["IBMSurfPart_2nd_faceFuncs_botSingCorner_largeRect.npz", \
#             "IBMSurfPart_3rd_tets_botSingCorner_largeRect.npz"]
# fileList = ["../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_1st_fineEdge_boxFix_topSingEdge_largeRect.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_2nd_topSingEdge_largeRect.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_2nd_faceFuncs_topSingEdge_largeRect.npz", \
#             "../IBMSurfPart_2nd_singLines_botSingEdge_largeRect.npz"]
# fileList = ["IBMSurfPart_1st_fineEdge_boxFix_topSingEdge_largeRect.npz", \
#             "IBMSurfPart_2nd_topSingEdge_largeRect.npz", \
#             "IBMSurfPart_2nd_faceFuncs_topSingEdge_largeRect.npz", \
#             "IBMSurfPart_3rd_splitTets_topSingEdge_largeRect.npz"]

# path = "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/heightTests/"
# fileList = ["IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_100nmTall_z=50nm.npz", \
#             "IBMSurfPart_1st_sideRect_zNorm_100nmTall_z=50nm.npz",
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_1umTall_z=500nm.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_2umTall_z=1um.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_4umTall_z=2um.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_6umTall_z=3um.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_8umTall_z=4um.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_10umTall_z=5um.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_100nmTall_z=50nm.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_1umTall_z=500nm.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_2umTall_z=1um.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_4umTall_z=2um.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_6umTall_z=3um.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_8umTall_z=4um.npz", \
#             "Prisms/IBMSurfPart_prisms_1st_sideRect_zNorm_10umTall_z=5um.npz"]
# fileList = ["IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_100nmTall_z=50nm.npz",
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_1umTall_z=50nm.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_2umTall_z=50nm.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_4umTall_z=50nm.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_6umTall_z=50nm.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_8umTall_z=50nm.npz", \
#             "IBMSurfPart_2nd_faceFuncs_sideRect_zNorm_10umTall_z=50nm.npz"]
# path = "../"
# fileList = ["IBMSurfPart_prisms_p=1,1_s=1,1_edgeFuncs.npz", "IBMSurfPart_prisms_p=1,1_s=2,1_edgeFuncs.npz", "IBMSurfPart_prisms_p=2,1_s=2,1_edgeFuncs.npz", \
#             "IBMSurfPart_prisms_10umMetal_p=1,1_s=1,1_edgeFuncs.npz", "IBMSurfPart_prisms_10umMetal_p=1,1_s=2,1_edgeFuncs.npz", "IBMSurfPart_prisms_10umMetal_p=2,1_s=2,1_edgeFuncs.npz"]
# fileList = ["groundedCPW_1st_MS_rect.npz", "groundedCPW_3rd_MS_rect_numSplit=1.npz", "groundedCPW_3rd_MS_rect_numSplit=5.npz", "groundedCPW_3rd_MS_rect_fineTrace_numSplit=1.npz", "groundedCPW_3rd_MS_rect_fineTrace_numSplit=5.npz"]

metalHeights=["100nm", "1um", "2um", "4um", "6um", "8um", "10um"]
path = "../Results/surfPart/"
fileList = [f"IBMSurfPart_metalHeight={metalHeight}.npz" for metalHeight in metalHeights]
fileList.extend([f"IBMSurfPart_prisms_metalHeight={metalHeight}.npz" for metalHeight in metalHeights])
fileList.extend([f"IBMSurfPart_prisms_metalHeight={metalHeight}_gradFuncs_s=2.npz" for metalHeight in metalHeights])
fileList.extend([f"IBMSurfPart_prisms_metalHeight={metalHeight}_gradFuncs_s=3.npz" for metalHeight in metalHeights])
fileList.extend([f"IBMSurfPart_prisms_metalHeight={metalHeights[0]}_gradFuncs_v2_s=2,1.npz"])
fileList.extend([f"IBMSurfPart_prisms_metalHeight={metalHeights[0]}_gradFuncs_v2_s=2,2.npz"])
fileList.extend([f"IBMSurfPart_prisms_metalHeight={metalHeights[0]}_gradFuncs_v2_s=3,2.npz"])

# path = "../Results/surfPart/"
# fileList = ["IBMSurfPart_plane_501points.npz"]
# fileList = ["IBMSurfPart_topCorner.npz"]

# path = "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/10umTrench/"
# fileList = ["IBMSurfPart_3rd_10umTrench_topSingCorner_largeRect.npz", \
#             "IBMSurfPart_3rd_10umTrench_botSingCorner_largeRect.npz",
#             "IBMSurfPart_3rd_10umTrench_botSingEdge_largeRect.npz", \
#             "IBMSurfPart_3rd_10umTrench_topSingEdge_largeRect.npz"]
            
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
  # for ii in range(3):
  #   mesh.point_data[f"file{fileInd}_{ii}"] = np.real(fields[:,ii]) + np.imag(fields[:,ii])
  mesh.point_data[filename] = np.real(fields) + np.imag(fields)
  
  # Also write in errors
  mesh.point_data[f"err_{filename}"] = (np.real(fields) + np.imag(fields) - np.real(origFields) - np.imag(origFields)) \
                                        / np.linalg.norm(np.real(origFields) - np.imag(origFields), axis=1).reshape(len(origFields),1)

# Save the vtk
vtkFile = f"fields_{fileList[0][:-4]}.vtk"
meshio.write(vtkFile, mesh)