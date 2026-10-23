import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit, minimize, basinhopping
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plots data along a set of singular lines using data from specific points
figList, axList = [[], []]

fileList = ["../../../3D-FEM-Results/Singular_Basis/Xmon/2D/Corner/soloXmon_tets_thickMetal_1st_E_triQuad.npz", \
            "../../../3D-FEM-Results/Singular_Basis/Xmon/2D/Corner/soloXmon_tets_thickMetal_2nd_singLines_E_triQuad.npz", \
            "../../../3D-FEM-Results/Singular_Basis/Xmon/2D/Corner/soloXmon_tets_thickMetal_3rd_singLines_E_faceFuncs_edgeFix.npz"]
# fileList = ["../../../3D-FEM-Results/Singular_Basis/Xmon/Volume_ints/lineVol/soloXmon_tets_thickMetal_1st_E.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Volume_ints/lineVol/soloXmon_tets_thickMetal_singLines_2nd_E_origMesh.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Volume_ints/lineVol/soloXmon_tets_thickMetal_singLines_2nd_E_faceFuncs_surfMesh.npz", \
#             "../soloXmon_tets_thickMetal_singLines_2nd_E_origMesh.npz", \
#             "../soloXmon_tets_thickMetal_singLines_surfFuncs_edgeFix_2nd_E_surfMesh.npz"]

# fileList = ["../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/IBMSurfPart_1st_fineEdge_topSingEdge.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/IBMSurfPart_2nd_topSingEdge.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/IBMSurfPart_2nd_surfFuncs_topSingEdge.npz"]
# fileList = ["../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/IBMSurfPart_1st_fineCorner_boxFix_botSingCorner_rect.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/IBMSurfPart_2nd_faceFuncs_botSingCorner_rect.npz"]

rates = []
Eana = []
fitColors = ['r', 'g', 'g']
for fileInd, filename in enumerate(fileList):
  meshData = np.load(filename, allow_pickle=True)
  points = meshData["fieldPoints"]
  vals = np.linalg.norm(meshData["pointFields"][0], axis=1)
  x, y = [points[:,0]*1e3, points[:,1]*1e3]
  # if fileInd == 0:
  vmin, vmax = [np.min(vals), np.max(vals)]

  curFig, curAx = plt.subplots()
  figList.append(curFig)
  axList.append(curAx)

  cntr = curAx.tricontourf(x, y, vals, vmin=vmin, vmax=vmax)
  curAx.scatter(x, y, c='r')
  cbar = curFig.colorbar(cntr)

  curAx.set_xlabel(rf"$x$ (mm)", fontfamily='Times New Roman', fontsize=14)
  curAx.set_ylabel(rf"$y$ (mm)", fontfamily='Times New Roman', fontsize=14)
  cbar.set_label(rf"$|E|$", fontfamily='Times New Roman', fontsize=14)


plt.show()