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

def singIntEval(coords, coeff, xLoc, yLoc, zLoc, basisOrder, singOrder):
  if zLoc.size == 1:
    loc = np.array([[xLoc, yLoc, zLoc]])
  else:
    loc = np.array([[xLoc, yLoc, zLoc[0,kk]] for kk in range(len(zLoc[0]))])
  fvals, _ = vecBasisSingular_orderN(coords, coeff, coeff, loc, basisOrder, singOrder)
  # fvals, _ = basis_vector_numba(coords, 0, 1, loc)

  # result = fvals[:,0]*fvals[:,0] # Nonsingular
  # result = fvals[0,:,0]*fvals[1,:,0] # (p,s) = (1,1)
  result = fvals[200,:,0]*fvals[201,:,0] # (p,s) = (2,3)
  # result = (fvals*fvals).reshape(fvals.shape[1], fvals.shape[0], fvals.shape[2]) # (p,s) = (2,3), all funcs
  if zLoc.size == 1:
    return result[0]
  else:
    return result.reshape(1,len(result))

curFunc = lambda x: x**2
testRes = tanhsinh(curFunc, 0, 1, minlevel=3)

def nestedInts(coords, coeff, basisOrder, singOrder):
  """
  Perform the nested integrals over the unit tetrahedron (z from 0 to 1-x-y, y from 0 to 1-x, x from 0 to 1)
  """
  def int_y(xvals):
    def int_z(yvals):
      if yvals.size == 1:
        yvals = yvals.reshape(1,1)

      zSum = np.zeros(yvals.shape)
      for jj, yval in enumerate(yvals[0]):
        if xval + yval == 1:
          # If the length of the integration region is 0, skip the integral
          zSum[0,jj] = 0
        else:
          zIntFunc = lambda z: singIntEval(coords, coeff, xval, yval, z, basisOrder, singOrder)
          curInt = tanhsinh(zIntFunc, 0, 1-xval-yval, minlevel=2, maxlevel=3, atol=1e-15, rtol=1e-6)
          if curInt.success == True:
            zSum[0,jj] = curInt.integral
          else:
            print(f"Integral failed for (x,y) = {(xval, yval)}")
          if np.isnan(curInt.integral):
            print(f"Integral is NaN for (x,y) = {(xval, yval)}")

      if yvals.size == 1:
        zSum = zSum.reshape(())

      return zSum
    
    if xvals.size == 1:
      xvals = xvals.reshape(1,1)

    ySum = np.zeros(xvals.shape)
    for ii, xval in enumerate(xvals[0]):
      if xval == 1:
        # If the length of the integration region is 0, skip the integral
        ySum[0,ii] = 0
      else:
        curInt = tanhsinh(int_z, 0, 1-xval, minlevel=2, maxlevel=3, atol=1e-15, rtol=1e-6)
        if curInt.success == True:
          ySum[0,ii] = curInt.integral
        else:
          print(f"Integral failed for x = {xval}")
        if np.isnan(curInt.integral):
          print(f"Integral is NaN for x = {xval}")

    if xvals.size == 1:
      ySum = ySum.reshape(())

    return ySum

  return tanhsinh(int_y, 0, 1, minlevel=2, maxlevel=3, atol=1e-15, rtol=1e-6)

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

intRes = nestedInts(coords, coeff, basisOrder, singOrder)

print(intRes)