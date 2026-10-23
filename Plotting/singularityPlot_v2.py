import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, basinhopping
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plots data along a set of singular lines using data from specific points
figList, axList = [[], []]

# fileList = ["../../../3D-FEM-Results/Singular_Basis/Xmon/Single_edge/1e-4_cylMesh/soloXmon_tets_thickMetal_1st_E.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Single_edge/224k_Mesh/soloXmon_tets_thickMetal_singLines_2nd_E.npz", \
#             "../soloXmon_tets_thickMetal_cutSub_v3_2nd_singLines_E_singleEdge.npz", \
#             "../soloXmon_tets_thickMetal_cutSub_v3_3rd_singLines_E_singleEdge.npz"]
# phi = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2]
# strList = ["1st order, 1.51 mil tets", r"2nd order singular, 166 tets", r"2nd", r"3rd"]

# fileList = ["../../../3D-FEM-Results/Singular_Basis/Xmon/Corner/soloXmon_tets_thickMetal_1st_E.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Corner/soloXmon_tets_thickMetal_2nd_singLines_E.npz",
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Corner/soloXmon_tets_thickMetal_2nd_singLines_E_faceFuncs.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Corner/soloXmon_tets_thickMetal_3rd_singLines_E_splitTets_noEdgeFuncs.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/Xmon/Corner/soloXmon_tets_thickMetal_3rd_singLines_E_splitTets.npz"]
# strList = ["1st order, 1.74 mil tets", r"2nd order singular", r"2nd order singular, edge functions", r"3rd order singular", r"3rd order singular, edge functions"]

# fileList = ["../../../3D-FEM-Results/Singular_Basis/Xmon/2D/Line/soloXmon_tets_thickMetal_1st_E.npz", \
#             # "../../../3D-FEM-Results/Singular_Basis/Xmon/2D/Line/soloXmon_tets_thickMetal_3rd_singLines_E_splitTets.npz", \
#             "../soloXmon_tets_thickMetal_cutSub_v3_3rd_singLines_E_xCurve.npz"]
# fileList = ["../soloXmon_tets_thickMetal_cutSub_fineXLine_1st_E_xCurve.npz", "../soloXmon_tets_thickMetal_cutSub_fineXLine_v2_1st_E_xCurve.npz", \


# fileList = ["../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_1st_fineEdge_boxFix_topSingEdge_lines.npz",
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_2nd_surfFuncs_topSingEdge_lines.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_3rd_splitTets_topSingEdge_lines.npz", \
#             "../IBMSurfPart_3rd_5splits_topSingEdge_lines.npz"]
# fileList = ["../IBMSurfPart_1st_10umTrench_fineEdgeRegion_vertLine_v2.npz", "../IBMSurfPart_3rd_10umTrench_vertLine.npz"]
# strList = ["1st order, 1.57 mil tets", r"2nd order singular, 11.3k tets", r"3rd order singular, 11.3k tets", "TEST"]
# fileList = ["../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_1st_fineEdge_boxFix_topSingEdge_lines.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/IBMSurfPart_3rd_tets_topSingEdge_lines.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/Prisms/IBMSurfPart_prisms_1st_topSingEdge_lines.npz"]
# strList = ["Non-singular", "Singular Tets, 3rd order", "Singular Prisms, 1st order"]
# fileList = ["../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/vertLine/IBMSurfPart_1st_nonSing_vertLine.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/vertLine/IBMSurfPart_3rd_tets_vertLine.npz", \
#             "../../../3D-FEM-Results/Singular_Basis/IBMSurfPart/noTrench/vertLine/IBMSurfPart_prisms_1st_vertLine.npz"]
# strList = ["Non-singular", "Singular Tets, 3rd order", "Singular Prisms, 1st order"]

# fileList = ["../groundedCPW_3rd_MS_line_numSplit=1.npz", "../groundedCPW_3rd_MS_line_numSplit=2.npz", \
#             "../groundedCPW_3rd_MS_line_numSplit=3.npz", "../groundedCPW_3rd_MS_line_numSplit=4.npz", \
#             "../groundedCPW_3rd_MS_line_numSplit=5.npz"]

# fileList = ["../groundedCPW_3rd_MS_line_numSplit=1_fineTrace.npz", "../groundedCPW_3rd_MS_line_numSplit=2_fineTrace.npz", \
#             "../groundedCPW_3rd_MS_line_numSplit=3_fineTrace.npz", "../groundedCPW_3rd_MS_line_numSplit=4_fineTrace.npz", \
#             "../groundedCPW_3rd_MS_line_numSplit=5_fineTrace.npz"]
# strList = ["numSplit=1", "numSplit=2", "numSplit=3", "numSplit=4", "numSplit=5"]




# fileList = ["../soloXmon_tets_thickMetal_cutSub_fineXLine_v2_2nd_E_xCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_1st_E_xCurve_5GHz_epr=11p45.npz", \
#             "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_2nd_E_xCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_3rd_E_xCurve_5GHz_epr=11p45.npz"]
# strList = [r"$(p,s) = (2,0), \, 597471 \, \mathrm{tets}$", r"$(p,s) = (2,1), \, 39 \, \mathrm{tets}$", r"$(p,s) = (2,2), \, 39 \, \mathrm{tets}$", r"$(p,s) = (2,3), \, 39 \, \mathrm{tets}$"]
fileList = ["../soloXmon_tets_thickMetal_cutSub_fineXLine_v2_2nd_E_xCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_p=1_s=3_xCurve_5GHz_epr=11p45.npz", \
            "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_p=2_s=3_xCurve_5GHz_epr=11p45.npz"]
strList = [r"$(p,s) = (2,0), \, 597471 \, \mathrm{tets}$", r"$(p,s) = (1,3), \, 39 \, \mathrm{tets}$", r"$(p,s) = (2,3), \, 39 \, \mathrm{tets}$"]
singEdgePos = np.array([2.88625000000000, 2.1071637061436, 0.030000])*1e-3 # X Line

# fileList = ["../soloXmon_tets_thickMetal_cutSub_fineXLine_v2_2nd_E_xCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_1st_E_xCurve_log_5GHz_epr=11p45.npz", \
#             "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_2nd_E_xCurve_log_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_3rd_E_xCurve_log_5GHz_epr=11p45.npz"]
# strList = [r"$(p,s) = (2,0), \, 597471 \, \mathrm{tets}$", r"$(p,s) = (2,1), \, 39 \, \mathrm{tets}$", r"$(p,s) = (2,2), \, 39 \, \mathrm{tets}$", r"$(p,s) = (2,3), \, 39 \, \mathrm{tets}$"]
# singEdgePos = np.array([2.88625000000000, 2.1071637061436, 0.030000])*1e-3 # X Line - Log

# fileList = ["../soloXmon_tets_thickMetal_cutSub_fineZLine_v2_2nd_E_zCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_1st_E_zCurve_5GHz_epr=11p45.npz", \
#             "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_2nd_E_zCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_3rd_E_zCurve_5GHz_epr=11p45.npz"]
# strList = [r"$(p,s) = (2,0), \, 618348 \, \mathrm{tets}$", r"$(p,s) = (2,1), \, 24 \, \mathrm{tets}$", r"$(p,s) = (2,2), \, 24 \, \mathrm{tets}$", r"$(p,s) = (2,3), \, 24 \, \mathrm{tets}$"]
# singEdgePos = np.array([2.96000000000000, 2.1071637061436, 0.015000])*1e-3 # Z Line

# fileList = ["../soloXmon_tets_thickMetal_cutSub_fineZLine_v2_2nd_E_zCurve_5GHz_epr=11p45.npz", "../soloXmon_tets_thickMetal_cutSub_cylinders_v5_singLines_3rd_E_zCurve_5GHz_epr=11p45.npz"]
# strList = [r"Standard Functions, $618348 \, \mathrm{tets}$", r"Singular Functions, $24 \, \mathrm{tets}$"]
# singEdgePos = np.array([2.96000000000000, 2.1071637061436, 0.015000])*1e-3 # Z Line - for graphical abstract





rates = []
Eana = []
fitColors = ['r', 'g', 'g']
# lineColors = ['k', 'r', 'b', 'm']
# colors = ['black', 'red', 'blue', 'orange']
# styles = ['-', '--', ':', '-.']
colors = ['black', 'blue', 'orange']
styles = ['-', ':', '-.']
lineErrs = [[] for _ in range(len(fileList)-1)]
for fileInd, filename in enumerate(fileList):
  meshData = np.load(filename, allow_pickle=True)
  if "eigvals" in meshData:
    ind = np.argmin(np.abs(meshData["eigvals"] - 25e9))
  else:
    ind = 0
  linePoints = [[meshData["fieldPoints"][0]]] # Split the points up by line
  lineFields = [[meshData["pointFields"][ind][0]]] # Same splitting for fields at those points
  pointDist = np.linalg.norm(meshData["fieldPoints"][1] - meshData["fieldPoints"][0])
  curLine = 0
  for ii in range(1, len(meshData["fieldPoints"])):
    if not np.any(np.isnan(meshData["pointFields"][ind][ii])):
      if np.linalg.norm(meshData["fieldPoints"][ii] - linePoints[curLine][-1]) > 10*pointDist:
        # If points are too far, assume this is the start of a new line
        curLine += 1
        linePoints.append([])
        lineFields.append([])
        # Remove the last point of the prior line to avoid errors there
        linePoints[curLine-1] = np.array(linePoints[curLine-1])[:-3,:]
        lineFields[curLine-1] = np.array(lineFields[curLine-1])[:-3,:]
      else:
        pointDist = np.linalg.norm(meshData["fieldPoints"][ii] - meshData["fieldPoints"][ii-1])
      linePoints[curLine].append(meshData["fieldPoints"][ii])
      lineFields[curLine].append(meshData["pointFields"][ind][ii])
  # Remove the last point of the prior line to avoid errors there
  linePoints[-1] = np.array(linePoints[-1])[:-3,:]
  lineFields[-1] = np.array(lineFields[-1])[:-3,:]

  # Renormalize the coordinate system to be centered about the position of the singularity
  linePoints -= singEdgePos

  if fileInd == 0:
    origLineFields = [np.array(lineFields[ii]) for ii in range(len(lineFields))] # Store the field values for the initial lines to calculate errors
    origLineInts = [np.linalg.norm(lineFields[ii]) for ii in range(len(lineFields))]
  # else:
  #   for ii in range(len(lineFields)):
  #     lineErrs[fileInd-1].append(np.linalg.norm(np.array(lineFields[ii]) - origLineFields[ii])/np.linalg.norm(origLineFields[ii]))
  #     # lineErrs[fileInd-1].append(np.linalg.norm(np.linalg.norm(lineFields[ii], axis=1) - np.linalg.norm(origLineFields[ii], axis=1))/np.linalg.norm(origLineFields[ii]))
  #     # lineErrs[fileInd-1].append((np.linalg.norm(lineFields[ii]) - origLineInts[ii]) / origLineInts[ii])
  #   print(f"Line errors for file {fileInd}: {lineErrs[fileInd-1]}")



  def singFunc(xdata, rate, sign, coeff, inter):
    return coeff*((-sign+1)/2 + sign*xdata)**(rate) + inter

  for line in range(len(linePoints)):
    if fileInd == 0:
      curFig, curAx = plt.subplots()
      figList.append(curFig)
      axList.append(curAx)
    else:
      curFig = figList[line]
      curAx = axList[line]

    curLinePoints = np.array(linePoints[line])
    lineDir = np.where(curLinePoints[1] - curLinePoints[0] != 0)[0][0]
    lineFieldsMag = np.linalg.norm(np.abs(lineFields[line]), axis=1)

    lineDist = np.linalg.norm(curLinePoints, axis=1)
    # fieldDir = 1
    # lineFieldsMag = np.imag(np.array(lineFields[line])[:,fieldDir]) + np.real(np.array(lineFields[line])[:,fieldDir])
    if "eigvals" in meshData: # Renormalize for eigenvalue problems
      lineFieldsMag /= np.max(np.abs(lineFieldsMag)) #np.linalg.norm(lineFieldsMag)
    # curAx.scatter(curLinePoints[:,lineDir]*1e6, lineFieldsMag/1e8, color=lineColors[fileInd], label=strList[fileInd])
    # curAx.plot(curLinePoints[:,lineDir]*1e6, lineFieldsMag/1e8, color=colors[fileInd], linestyle=styles[fileInd], label=strList[fileInd], linewidth=2)
    if lineDist[0] > 0:
      curAx.plot(lineDist*1e6, lineFieldsMag/1e8, color=colors[fileInd], linestyle=styles[fileInd], label=strList[fileInd], linewidth=2)
    else:
      curAx.plot(-lineDist*1e6, lineFieldsMag[::-1]/1e8, color=colors[fileInd], linestyle=styles[fileInd], label=strList[fileInd], linewidth=2)
    # curAx.scatter(curLinePoints[:,lineDir]*1e3, np.imag(np.array(lineFields[line])[:,2]), label=strList[fileInd])

    ## Perform an polynomial fit
    # Normalize to improve the fit
    xOff = 1e-6
    # if "singLines" in filename: # If this data is singular, isolate the singular element
    #   endSingPoint = -1
    #   prevField = lineFieldsMag[-1]
    #   for ind, fieldVal in enumerate(lineFieldsMag[-2:0:-1]):
    #     if np.abs(fieldVal - prevField) < np.abs(fieldVal)*1e-6:
    #       endSingPoint = len(lineFieldsMag) - ind
    #       break
    #     else:
    #       prevField = fieldVal
    #   fitCoords = curLinePoints[endSingPoint:,lineDir]
    #   fitFields = lineFieldsMag[endSingPoint:]
    # else:
    fitCoords = curLinePoints[:,lineDir]
    fitFields = lineFieldsMag

    normCoords = (fitCoords - np.min(fitCoords)) / (np.max(fitCoords) - np.min(fitCoords)) + xOff
    normFields = fitFields / np.max(np.abs(fitFields))

    # Form initial estimates
    # midVal = (np.max(normFields) + np.min(normFields))/2 # Midpoint between the max and min
    if "singLines" not in filename:
      if normCoords[1] - normCoords[0] > 0: # Coordinates from left to right
        coeff = np.sign(normFields[0])
        sign = -1#np.sign(normFields[0])
        p0 = np.array([-1/3, sign, coeff, normFields[-1]])
        normCoords -= 2*xOff
      else: # Coordinates from right to left
        coeff = np.sign(normFields[0]) # 1/(np.mi(scaledBotCoords)**(rate))
        sign = 1#np.sign(normFields[0])
        p0 = np.array([-1/3, sign, coeff, normFields[0]])
        

      bounds = ([-1.1, -1, -2, -1], [1.1, 1, 2, 1])
      optP, _ = curve_fit(singFunc, normCoords, normFields, p0, bounds=bounds)
      fit = singFunc(normCoords, *optP) * np.max(np.abs(fitFields))
      # curAx.plot(fitCoords*1e3, fit, fitColors[fileInd], label=rf"$\nu = {optP[0]:.3f}$")
      rates.append(optP[0])

    # if fileInd == 0:
    #   # Add the analytical curve
    #   rho = np.linalg.norm(curLinePoints - curLinePoints[-1] - (curLinePoints[-1] - curLinePoints[-2]), axis=1) # Radial distance from singularity
    #   coeff = 2/3
    #   numTerms = 20 # Number of terms in the summation
    #   Erho_single = -coeff*rho**(coeff-1)*np.sin(coeff*phi[line])
    #   Ephi_single = -coeff*rho**(coeff-1)*np.cos(coeff*phi[line])
    #   Erho = np.sum(np.array([-jj*coeff*rho**(jj*coeff-1)*np.sin(jj*coeff*phi[line]) for jj in range(1,numTerms+1)]), axis=0)
    #   Ephi = np.sum(np.array([-jj*coeff*rho**(jj*coeff-1)*np.cos(jj*coeff*phi[line]) for jj in range(1,numTerms+1)]), axis=0)

    #   Etot_single = np.array([np.cos(phi[line])*Erho_single - np.sin(phi[line])*Ephi_single, np.sin(phi[line])*Erho_single + np.cos(phi[line])*Ephi_single, np.zeros((len(rho),))]).T
    #   EtotMag_single = np.linalg.norm(Etot_single, axis=1)
    #   Etot = np.array([np.cos(phi[line])*Erho - np.sin(phi[line])*Ephi, np.sin(phi[line])*Erho + np.cos(phi[line])*Ephi, np.zeros((len(rho),))]).T
    #   EtotMag = np.linalg.norm(Etot, axis=1)

    #   scalFactor = lineFieldsMag[0]/EtotMag_single[0]
    #   EtotMag_single *= scalFactor
    #   EtotMag *= scalFactor

    #   curAx.plot(fitCoords*1e3, EtotMag_single, "-k", label=rf"Analytical, 1 term")
    #   curAx.plot(fitCoords*1e3, EtotMag, "--g", label=rf"Analytical, {numTerms} terms")
    #   Eana.append(EtotMag)

    dirStr = ["x", "y", "z"]
    # curAx.set_xlabel(rf"${dirStr[lineDir]} \, (\mu$"r"$\mathrm{m})$", fontfamily='Times New Roman', fontsize=24)
    curAx.set_xlabel(rf"Distance from Singular Edge $(\mu$"r"$\mathrm{m})$", fontfamily='Times New Roman', fontsize=22)
    curAx.set_ylabel(rf"$|E|$ (a.u.)", fontfamily='Times New Roman', fontsize=22)
    # curAx.set_ylabel(rf"$E_{dirStr[fieldDir]}$", fontfamily='Times New Roman', fontsize=14)
print(f"Polynomial coefficients for each line: {rates}")

for ii, curAx in enumerate(axList):
  curAx.legend(prop={'family': 'Times New Roman', 'size':22})
  curAx.tick_params(axis='both', which='major', labelsize=22, labelfontfamily='Times New Roman')
  curAx.grid()
  # figList[ii].savefig(f"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Papers/My Papers/SBFs 2026/Images/Xmon/xmon_zLine_{ii}_ga.png", dpi=400, bbox_inches="tight", pad_inches=0.0)
  figList[ii].savefig(f"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Papers/My Papers/SBFs 2026/MR1/Images/Xmon/xmon_xLine_{ii}_p.png", dpi=400, bbox_inches="tight", pad_inches=0.0)
plt.show()



## Analytically fit the data from each line to an expansion of the singular behavior
filename = fileList[0]
meshData = np.load(filename, allow_pickle=True)
linePoints = [[meshData["fieldPoints"][0]]] # Split the points up by line
lineBasisFields = [[meshData["pointBasisFields"][:,0,:]]] # Same splitting for fields at those points
curLine = 0
pointDist = np.linalg.norm(meshData["fieldPoints"][1] - meshData["fieldPoints"][0])
for ii in range(1, len(meshData["fieldPoints"])):
  if np.linalg.norm(meshData["fieldPoints"][ii] - linePoints[curLine][-1]) > 10*pointDist:
    # If points are too far, assume this is the start of a new line
    curLine += 1
    linePoints.append([])
    lineBasisFields.append([])
  linePoints[curLine].append(meshData["fieldPoints"][ii])
  lineBasisFields[curLine].append(meshData["pointBasisFields"][:,ii])

numLines = len(linePoints)
phiVals = np.zeros((numLines,), dtype=np.float64)
rhoVals = np.zeros((numLines, len(linePoints[0])), dtype=np.float64)
fieldMags = np.zeros((numLines, len(linePoints[0])), dtype=np.float64)
for ii in range(numLines):
  curLinePoints = np.array(linePoints[ii])
  rhoVals[ii] = np.linalg.norm(curLinePoints - curLinePoints[-1] - (curLinePoints[-1] - curLinePoints[-2]), axis=1) # Radial distance from singularity
  lineDir = linePoints[ii][1] - linePoints[ii][0]
  lineDir = lineDir / np.linalg.norm(lineDir) # Unit vector of the integration line
  phiVals[ii] = np.arctan2(lineDir[1], lineDir[0]) # Polar angle of the integration line
  if phiVals[ii] < 0: # Limit phi to be on [0, 2*pi)
    phiVals[ii] += 2*np.pi
  fieldMags[ii] = np.linalg.norm(lineFields[ii], axis=1)

# Optimize coefficients of the expansion around the form of each curve
def optSingExpansion(rhoVals, phiVals, fieldMags, numTerms, nu, coeff_scale, coeffs):
  cost = 0
  for ii in range(len(phiVals)):
    Erho = np.sum(np.array([-jj*coeff_scale[jj-1]*coeffs[jj-1]*nu*rhoVals[ii]**(jj*nu-1)*np.sin(jj*nu*phiVals[ii]) for jj in range(1,numTerms+1)]), axis=0)
    Ephi = np.sum(np.array([-jj*coeff_scale[jj-1]*coeffs[jj-1]*nu*rhoVals[ii]**(jj*nu-1)*np.cos(jj*nu*phiVals[ii]) for jj in range(1,numTerms+1)]), axis=0)
    Etot = np.array([np.cos(phiVals[ii])*Erho - np.sin(phiVals[ii])*Ephi, np.sin(phiVals[ii])*Erho + np.cos(phiVals[ii])*Ephi, np.zeros((len(rhoVals[ii]),))]).T
    EtotMag = np.linalg.norm(Etot, axis=1)

    cost += np.linalg.norm(EtotMag - fieldMags[ii])

  return cost


numTerms = 5 # Number of terms to use in the expansion
nu = 2/3
yScale = fieldMags[0,0]
coeff_scale = np.array([1/(nu*rhoVals[0,0]**(nu-1))**(-(2*ii)+1) for ii in range(numTerms)], dtype=np.float64)
x0 = np.ones((numTerms,), dtype=np.float64)
optFunc = lambda x: optSingExpansion(rhoVals, phiVals, fieldMags/yScale, numTerms, nu, coeff_scale, x)
# optRes = minimize(optFunc, x0, method="L-BFGS-B")
optRes = basinhopping(optFunc, x0, stepsize=1e5, minimizer_kwargs={"method":"L-BFGS-B"}, disp=True)
coeffs = optRes["x"]*yScale*coeff_scale

cost_init = optFunc(x0)
print(f"Final cost function value: {optRes.fun}. Initial cost function value: {cost_init}")
print(f"Coefficients for analytical fit: {coeffs}")
print(f"Initial coefficients: {x0*yScale*coeff_scale}")

for ii, curAx in enumerate(axList):
  Erho = np.sum(np.array([-jj*coeffs[jj-1]*nu*rhoVals[ii]**(jj*nu-1)*np.sin(jj*nu*phiVals[ii]) for jj in range(1,numTerms+1)]), axis=0)
  Ephi = np.sum(np.array([-jj*coeffs[jj-1]*nu*rhoVals[ii]**(jj*nu-1)*np.cos(jj*nu*phiVals[ii]) for jj in range(1,numTerms+1)]), axis=0)
  Etot = np.array([np.cos(phiVals[ii])*Erho - np.sin(phiVals[ii])*Ephi, np.sin(phiVals[ii])*Erho + np.cos(phiVals[ii])*Ephi, np.zeros((len(rhoVals[ii]),))]).T
  EtotMag = np.linalg.norm(Etot, axis=1)

  curLinePoints = np.array(linePoints[ii])
  lineDir = np.where(curLinePoints[1] - curLinePoints[0] != 0)[0][0]
  curAx.plot(curLinePoints[:,lineDir]*1e3, EtotMag, "-k", label=rf"Analytical, {numTerms} terms")

  curAx.legend(prop={'family': 'Times New Roman', 'size':14})
  curAx.tick_params(axis='both', which='major', labelsize=12, labelfontfamily='Times New Roman')
  curAx.grid()



plt.show()




# ## Plot the contribution of each basis function to one of the plots for the singular element
# filename = fileList[1]
# meshData = np.load(filename, allow_pickle=True)
# linePoints = [[meshData["fieldPoints"][0]]] # Split the points up by line
# lineBasisFields = [[meshData["pointBasisFields"][:,0,:]]] # Same splitting for fields at those points
# curLine = 0
# for ii in range(1, len(meshData["fieldPoints"])):
#   if np.linalg.norm(meshData["fieldPoints"][ii] - linePoints[curLine][-1]) > 1e-6:
#     # If points are too far, assume this is the start of a new line
#     curLine += 1
#     linePoints.append([])
#     lineBasisFields.append([])
#   linePoints[curLine].append(meshData["fieldPoints"][ii])
#   lineBasisFields[curLine].append(meshData["pointBasisFields"][:,ii])

# # Plot the contributions of each basis function to one of the integration lines
# lineInd = 1
# linePoints = np.array(linePoints[lineInd])
# lineDir = np.where(linePoints[1] - linePoints[0] != 0)[0][0]

# lineBasisFields = lineBasisFields[lineInd]
# numBasis = len(lineBasisFields[0]) # Number of basis functions

# curFig, curAx = plt.subplots()
# fieldDir = 0
# for basisInd in range(9,numBasis):
#   # lineFieldsMag = np.linalg.norm(np.abs([lineBasisFields[kk][basisInd] for kk in range(len(lineBasisFields))]), axis=1)
#   lineFieldsMag = [np.imag(lineBasisFields[kk][basisInd][fieldDir]) for kk in range(len(lineBasisFields))]
#   curAx.scatter(linePoints[:,lineDir]*1e3, lineFieldsMag, label=f"Basis Function {basisInd}")
# # totFieldMag = np.linalg.norm(np.abs(np.sum(lineBasisFields, axis=1)), axis=1)
# totFieldMag = np.sum(np.imag(np.array(lineBasisFields)[:,:,fieldDir]), axis=1)
# curAx.scatter(linePoints[:,lineDir]*1e3, totFieldMag, label=f"Total field")

# dirStr = ["x", "y", "z"]
# curAx.set_xlabel(rf"${dirStr[lineDir]}$ (mm)", fontfamily='Times New Roman', fontsize=14)
# curAx.set_ylabel(rf"$E_{dirStr[fieldDir]}$", fontfamily='Times New Roman', fontsize=14)
# curAx.legend(prop={'family': 'Times New Roman', 'size':14})
# curAx.tick_params(axis='both', which='major', labelsize=12, labelfontfamily='Times New Roman')
# curAx.grid()

# plt.show()