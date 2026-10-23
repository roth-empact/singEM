import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

meshData = np.load("soloXmon_lineTets.npz", allow_pickle=True)
fieldData = np.load("fields_soloXmon_tets_thickMetal.npy")
dirs = [0, 0, 0, 1, 1, 1] # Directions of each curve
figList, axList = [[], []]

def expFunc(xdata, rate, coeff, inter):
  return coeff*xdata**(rate) + inter
def doubleExpFunc(xdata, rate1, coeff1, rate2, coeff2, inter):
  return coeff1*(xdata - np.min(xdata))**(rate1) + coeff2*(np.max(xdata)-xdata)**(rate2) + inter

botRates, topRates = [[], []]
for ii, line in enumerate(meshData["lineNames"]):
  curFig, curAx = plt.subplots()
  figList.append(curFig)
  axList.append(curAx)

  tetDirCoords = meshData["tetCenterCoords"].item()[line][:,dirs[ii]]
  tetInds = meshData["tetInds"].item()[line]
  tetFields = np.linalg.norm(fieldData[tetInds, 1:], axis=1)
  curAx.scatter(tetDirCoords*1e3, tetFields, label=rf"Measured tet fields")

  ## Split the data in half and perform an exponential fit on each side
  # Normalize to improve the fit
  normCoords = (tetDirCoords - np.min(tetDirCoords)) / (np.max(tetDirCoords) - np.min(tetDirCoords))
  botInds = np.where(normCoords <= 0.5)[0]
  topInds = np.where(normCoords > 0.5)[0]
  normFields = tetFields / np.max(tetFields)

  ## Perform an exponential fit of the data at each end
  maxDist = 5e-6 # Maximum distance from the singularity to perform the fit along
  botInds = np.where(tetDirCoords - np.min(tetDirCoords) <= maxDist)[0]
  topInds = np.where(np.max(tetDirCoords) - tetDirCoords <= maxDist)[0]

  xOff = 0 # Offset in the scaled coordinate system
  scaledBotCoords = (tetDirCoords[botInds] - np.min(tetDirCoords)) / (np.max(tetDirCoords[botInds]) - np.min(tetDirCoords)) + xOff
  scaledTopCoords = (np.max(tetDirCoords) - tetDirCoords[topInds]) / (np.max(tetDirCoords) - np.min(tetDirCoords[topInds])) + xOff
  # Ignore the points particularly close to the singularity to avoid picking up innaccuracy there
  botInds = botInds[scaledBotCoords >= 0.1]
  scaledBotCoords = (tetDirCoords[botInds] - np.min(tetDirCoords)) / (np.max(tetDirCoords[botInds]) - np.min(tetDirCoords)) + xOff
  topInds = topInds[scaledTopCoords >= 0.1]
  scaledTopCoords = (np.max(tetDirCoords) - tetDirCoords[topInds]) / (np.max(tetDirCoords) - np.min(tetDirCoords[topInds])) + xOff

  # Form initial estimates
  botCoordSort = np.argsort(scaledBotCoords)
  midVal = (np.max(normFields) + np.min(normFields))/2 # Midpoint between the max and min
  if normFields[botInds[botCoordSort][0]] < midVal: # Below mean -> zero singularity
    p0Bot = np.array([1.0, normFields[botInds[botCoordSort][-1]], 0])
  else: # Above mean -> infinite singularity
    rate = -1/3
    coeff = 1/(np.min(scaledBotCoords)**(rate))
    yint = np.max([1e-5, normFields[botInds[botCoordSort][-1]] - coeff*np.max(scaledBotCoords)**(rate)])
    p0Bot = np.array([rate, coeff, yint])

  topCoordSort = np.argsort(scaledTopCoords)
  if normFields[topInds[topCoordSort][0]] < midVal: # Below mean -> zero singularity
    p0Top = np.array([1.0, normFields[topInds[topCoordSort][-1]], 0])
  else: # Above mean -> infinite singularity
    rate = -1/3
    coeff = 1/(np.min(scaledTopCoords)**(rate))
    yint = np.max([1e-5, normFields[topInds[topCoordSort][-1]] - coeff*np.max(scaledTopCoords)**(rate)])
    p0Top = np.array([rate, coeff, yint])


  bounds = ([-1.1, -2, 0], [1.1, 2, 1])
  botOptP, _ = curve_fit(expFunc, scaledBotCoords, normFields[botInds], p0Bot, bounds=bounds)
  topOptP, _ = curve_fit(expFunc, scaledTopCoords, normFields[topInds], p0Top, bounds=bounds)
  botFit = expFunc(scaledBotCoords, *botOptP) * np.max(tetFields)
  topFit = expFunc(scaledTopCoords, *topOptP) * np.max(tetFields)
  totFitCoords = np.append(tetDirCoords[botInds], tetDirCoords[topInds])
  # totFit = np.append(botFit, topFit)
  # coordSort = np.argsort(totFitCoords)
  # curAx.plot(totFitCoords[coordSort]*1e3, totFit[coordSort], '-r', label=rf"Exponential fit")
  curAx.plot(tetDirCoords[botInds][botCoordSort]*1e3, botFit[botCoordSort], '-r', label=rf"Exponential fit")
  curAx.plot(tetDirCoords[topInds][topCoordSort]*1e3, topFit[topCoordSort], '-r')

  botRates.append(botOptP[0])
  topRates.append(topOptP[0])

  curAx.grid()
  if dirs[ii] == 0:
    curAx.set_xlabel(r"$x$ (mm)", fontfamily='Times New Roman', fontsize=14)
  elif dirs[ii] == 1:
    curAx.set_xlabel(r"$y$ (mm)", fontfamily='Times New Roman', fontsize=14)
  curAx.set_ylabel(r"$|E|$", fontfamily='Times New Roman', fontsize=14)
  curAx.legend(prop={'family': 'Times New Roman', 'size':14})
  curAx.tick_params(axis='both', which='major', labelsize=12, labelfontfamily='Times New Roman')


print(f"Exponential rates for each line:")
for ii in range(len(meshData["lineNames"])):
  print(f"Singular coefficients for line {ii}: Left side: {botRates[ii]:.3f}. Right side: {topRates[ii]:.3f}")

plt.show()