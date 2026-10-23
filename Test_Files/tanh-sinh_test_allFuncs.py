import numpy as np
from scipy.integrate import tanhsinh
import sys
sys.path.append("./")
from singBasis_tets import vecBasisSingular_orderN
from basis_funcs import basis_vector_numba

## Testing tanh-sinh quadrature for computing integrals over triangles / tets

# Start by trying to integrate over a tet
coords = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]], dtype=np.float64)
# origTriCoords = coords[[0,1,3]]#coords[:3]
# normVec = np.cross(origTriCoords[0] - origTriCoords[1], origTriCoords[0] - origTriCoords[2])
# normVec /= np.linalg.norm(origTriCoords)
# triCoords, rotMat = rotateCoords2D(origTriCoords, normVec, True)
basisOrder, singOrder = [2,3]
coeff = 1/2

def singIntEval(coords, coeff, xLoc, yLoc, zLoc, basisOrder, singOrder, mnDirs, thetaVals):
  if len(zLoc.shape) == 3: # Special case when only one point is provided
    loc = np.array([[xLoc, yLoc, zLoc[0,0,0]]])
  else:
    loc = np.array([[xLoc, yLoc, zLoc[0,0,0,kk]] for kk in range(len(zLoc[0,0,0]))])
  fvals, _ = vecBasisSingular_orderN(coords, coeff, coeff, loc, basisOrder, singOrder)
  # fvals, _ = basis_vector_numba(coords, 0, 1, loc)

  result = np.zeros((fvals.shape[0], fvals.shape[0], len(mnDirs), fvals.shape[1]), dtype=np.float64)
  for ii in range(fvals.shape[1]):
    for mn in range(6):
      result[:,:,mn,ii] = thetaVals[mn] * np.outer(fvals[:,ii,mnDirs[mn,0]], fvals[:,ii,mnDirs[mn,0]])

  # result = np.zeros((len(mnDirs), fvals.shape[1]), dtype=np.float64)
  # for ii in range(fvals.shape[1]):
  #   for mn in range(6):
  #     result[mn,ii] = np.outer(fvals[200,ii,mnDirs[mn,0]], fvals[201,ii,mnDirs[mn,0]])

  if len(zLoc.shape) == 3:
    return result.reshape(tuple([result.shape[jj] for jj in range(len(result.shape)-1)]))
  else:
    return result

curFunc = lambda x: x**2
testRes = tanhsinh(curFunc, 0, 1, minlevel=3)

def nestedInts(coords, coeff, basisOrder, singOrder, numFuncs, mnDirs, thetaVals):
  """
  Perform the nested integrals over the unit tetrahedron (z from 0 to 1-x-y, y from 0 to 1-x, x from 0 to 1)
  """
  def int_y(xvals):
    def int_z(yvals):
      zSum = np.zeros(yvals.shape)
      if len(yvals.shape) == 3:
        yvals = yvals[0,0,0].reshape(1,)
      else:
        yvals = yvals[0,0,0,:]

      for jj, yval in enumerate(yvals):
        if xval + yval == 1:
          # If the length of the integration region is 0, skip the integral
          zSum[0,jj] = 0
        else:
          zIntFunc = lambda z: singIntEval(coords, coeff, xval, yval, z, basisOrder, singOrder, mnDirs, thetaVals)
          # The bounds needs to be set as vectors to make sure the bounds are correct for all terms
          curInt = tanhsinh(zIntFunc, lowerBound, 1-xval-yval, minlevel=2, maxlevel=4, atol=1e-10, rtol=1e-6, preserve_shape=True)
          if np.all(curInt.success) == True:
            if len(zSum.shape) == 3:
              zSum = curInt.integral
            else:
              zSum[:,:,:,jj] = curInt.integral
          else:
            print(f"Integral failed for (x,y) = {(xval, yval)}")
          if np.any(np.isnan(curInt.integral)):
            print(f"Integral is NaN for (x,y) = {(xval, yval)}")

      # if yvals.size == 1:
      #   zSum = zSum.reshape(())

      return zSum
    
    ySum = np.zeros(xvals.shape)
    if len(xvals.shape) == 3:
      xvals = xvals[0,0,0].reshape(1,)
    else:
      xvals = xvals[0,0,0,:]

    for ii, xval in enumerate(xvals):
      if xval == 1:
        # If the length of the integration region is 0, skip the integral
        ySum[0,ii] = 0
      else:
        curInt = tanhsinh(int_z, lowerBound, 1-xval, minlevel=2, maxlevel=4, atol=1e-15, rtol=1e-6, preserve_shape=True)
        if np.all(curInt.success) == True:
          ySum = curInt.integral
        else:
          print(f"Integral failed for x = {xval}")
        if np.any(np.isnan(curInt.integral)):
          print(f"Integral is NaN for x = {xval}")

    # if xvals.size == 1:
    #   ySum = ySum.reshape(())

    return ySum

  # The bounds needs to be set as vectors to make sure the bounds are correct for all terms
  lowerBound = np.zeros((numFuncs,numFuncs,len(mnDirs)), dtype=np.float64) 
  return tanhsinh(int_y, lowerBound, 1, minlevel=2, maxlevel=4, atol=1e-15, rtol=1e-6, preserve_shape=True)

# def nestedInts(coords, coeff, basisOrder, singOrder):
#   """
#   Perform the nested integrals over the unit tetrahedron (z from 0 to 1-x-y, y from 0 to 1-x, x from 0 to 1)
#   """
#   def int_y(xval):
#     def int_z(yval):
#       zIntFunc = lambda z: singIntEval(coords, coeff, xval, yval, z, basisOrder, singOrder)
#       return tanhsinh(zIntFunc, 0, 1-xval-yval).integral
#     return tanhsinh(int_z, 0, 1-xval).integral
#   return tanhsinh(int_y, 0, 1, minlevel=3)

mnDirs = np.array([[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]], dtype=np.int32)
thetaVals = np.zeros((len(mnDirs),), dtype=np.float64)
for mn in range(len(mnDirs)):
  if mnDirs[mn][0] == mnDirs[mn][1]:
    thetaVals[mn] = 1/2
  else:
    thetaVals[mn] = 1

testPt = 0.25*np.ones((1,3), dtype=np.float64)
fvals_temp, _ = vecBasisSingular_orderN(coords, coeff, coeff, testPt, basisOrder, singOrder)
numFuncs = fvals_temp.shape[0]
intRes = nestedInts(coords, coeff, basisOrder, singOrder, numFuncs, mnDirs, thetaVals)

print(intRes)