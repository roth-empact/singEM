"""
   Copyright 2026 Samuel Elkin, Ghazi Khan, Thomas Roth

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import sys
import petsc4py #type:ignore
petsc4py.init(sys.argv)
from petsc4py import PETSc #type:ignore
import sys, time

def buildFDMat(matDict, freq, simType, staticFlag=False):
  """
  Builds the full frequency domain system.

  Inputs:
    matDict    - Dictionary containing each of the matrices as PETSc "Mat" objects
    freq       - The frequency to build the system at
    simType    - The type of simulation being used ("E" or "A")
    staticFlag - If "True" form a static version of the system to be used as a preconditioner
  Outputs:
    A          - The full system (or static system if staticFlag=True)
    A_diagInv  - Vector used for diagonal scaling of the matrix
  """

  omega = 2*np.pi*freq

  ## Set up the matrix and apply diagonal scaling
  if not staticFlag:
    vecMat = matDict["K2"].copy()
    vecMat.scale(-omega**2)
    vecMat.axpy(1.0, matDict["K1"], PETSc.Mat.Structure.SUBSET)
    vecMat.axpy(-1j*omega, matDict["Ree"], PETSc.Mat.Structure.DIFFERENT)
    vecMat.axpy(-1j*omega, matDict["Bee"], PETSc.Mat.Structure.SUBSET)
    vecMat.axpy(-omega**2, matDict["Tee"], PETSc.Mat.Structure.DIFFERENT)
  else:
    vecMat = matDict["K1"].copy()
  vecMat.axpy(1.0, matDict["See"], PETSc.Mat.Structure.DIFFERENT)
  if simType == "E": ## TODO: For A-phi, remember to set preconditioner re-use if using static
    A = vecMat
    vecIS, scalIS = [[], []]
  elif simType == "A": ## TODO: Add scalaing factors (mu0, ep0)
    ep0, mu0 = [1e-9 / (36 * np.pi), 4e-7 * np.pi]
    vecMat.scale(mu0)

    scalMat = matDict["Gnn"].copy()
    scalMat.scale(-omega**2)
    if not staticFlag:
      scalMat.axpy(-1j*omega, matDict["Bnn"], PETSc.Mat.Structure.DIFFERENT)
    # scalDiag = scalMat.getDiagonal()
    # # scalDiagInv = (scalDiag.sqrtabs()).reciprocal()
    # scalDiagInv_np = np.reciprocal(np.sqrt(np.abs(scalDiag.array)))
    # scalDiagInv = scalMat.createVecLeft()
    # ownRange = scalDiagInv.owner_range
    # scalDiagInv.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), scalDiagInv_np)
    # scalDiagInv.assemblyBegin()
    # scalDiagInv.assemblyEnd()
    # scalMat.diagonalScale(scalDiagInv, scalDiagInv)

    vecScalMat = matDict["Ken"].copy()
    vecScalMat.scale(-1j*omega)
    if not staticFlag:
      vecScalMat.axpy(-1j*omega, matDict["Ten"], PETSc.Mat.Structure.DIFFERENT)
      vecScalMat.axpy(1, matDict["Ren"], PETSc.Mat.Structure.DIFFERENT)
      vecScalMat.axpy(-1/(1j*omega), matDict["Sen"], PETSc.Mat.Structure.DIFFERENT)
    vecScalMat.scale(1/ep0)
    # vecScalMat.diagonalScale(vecDiagInv, scalDiagInv)

    scalVecMat = matDict["Kne"].copy()
    scalVecMat.scale(-1j*2*np.pi*freq*mu0*ep0)
    # scalVecMat.diagonalScale(scalDiagInv, vecDiagInv)

    A = PETSc.Mat().createNest([[vecMat, vecScalMat], [scalVecMat, scalMat]], comm=comm)
    # A = PETSc.Mat().createNest([[scalMat, scalVecMat], [vecScalMat, vecMat]], comm=comm)
    [[vecIS, _], [_, scalIS]] = A.getNestISs()
    A.convert("aij")

  ## Apply diagonal scaling
  A_diag = A.getDiagonal()
  A_diagInv_np = np.reciprocal(np.sqrt(np.abs(A_diag.array)))
  A_diagInv = A.createVecLeft()
  ownRange = A_diagInv.owner_range
  A_diagInv.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), A_diagInv_np)
  A_diagInv.assemblyBegin()
  A_diagInv.assemblyEnd()
  A.diagonalScale(A_diagInv, A_diagInv)

  return A, A_diagInv, vecIS, scalIS

def buildTDMats(matDict, dt, beta, simType):
  """
  Builds matrices for time-marching

  Inputs:
    matDict    - Dictionary containing each of the matrices as PETSc "Mat" objects
    dt         - Timestep size
    beta       - Newmark-beta constant
    simType    - The type of simulation being used ("E" or "A")
  Outputs:
    advMat         - Matrix for the vector advanced in time (n+1)
    curMat         - Matrix for the vector current in time (n)
    preMat         - Matrix for the previous vector in time (n-1)
    advMat_diagInv - Vector used for diagonal scaling
  """

  # Build the T, R, and S matrices
  Tee = matDict["K2"].copy()
  Tee.axpy(1.0, matDict["Tee"], PETSc.Mat.Structure.DIFFERENT)
  Ree = matDict["Ree"].copy()
  Ree.axpy(1.0, matDict["Bee"], PETSc.Mat.Structure.DIFFERENT)
  See = matDict["K1"].copy()
  See.axpy(1.0, matDict["See"], PETSc.Mat.Structure.DIFFERENT) # TODO: Add integral for inductances
  if simType == "E":
    T, R, S = [Tee, Ree, See]
  elif simType == "A":
    ep0, mu0 = [1e-9 / (36 * np.pi), 4e-7 * np.pi]

    Tee.scale(mu0)
    Ten = PETSc.Mat().create(comm=comm)
    Ten.setSizes(matDict["Ken"].getSizes())
    Tne = PETSc.Mat().create(comm=comm)
    Tne.setSizes(matDict["Kne"].getSizes())
    Tnn = matDict["Gnn"]
    T = PETSc.Mat().createNest([[Tee, Ten], [Tne, Tnn]], comm=comm)
    T.convert("aij")

    Ree.scale(mu0)
    Ren = matDict["Ken"].copy()
    Ren.axpy(1.0, matDict["Ten"], PETSc.Mat.Structure.DIFFERENT)
    Ren.scale(1/ep0)
    Rne = matDict["Kne"].copy()
    Rne.scale(mu0*ep0)
    Rnn = matDict["Bnn"].copy()
    R = PETSc.Mat().createNest([[Ree, Ren], [Rne, Rnn]], comm=comm)
    R.convert("aij")

    See.scale(mu0)
    Sen = matDict["Ren"].copy()
    Sne = PETSc.Mat().create(comm=comm)
    Sne.setSizes(matDict["Kne"].getSizes())
    Snn = PETSc.Mat().create(comm=comm)
    Snn.setSizes(matDict["Gnn"].getSizes())
    S = PETSc.Mat().createNest([[See, Sen], [Sne, Snn]], comm=comm)
    S.convert("aij")

  # Build the matrices for each time coefficient
  T.scale(1/dt**2)
  advMat = T.copy()
  advMat.axpy(2/dt, R, PETSc.Mat.Structure.DIFFERENT)
  advMat.axpy(beta, S, PETSc.Mat.Structure.DIFFERENT)
  curMat = T.copy()
  curMat.scale(2)
  curMat.axpy(-(1-2*beta), S, PETSc.Mat.Structure.DIFFERENT)
  preMat = T.copy()
  preMat.scale(-1)
  preMat.axpy(2/dt, R, PETSc.Mat.Structure.DIFFERENT)
  preMat.axpy(-beta, S, PETSc.Mat.Structure.DIFFERENT)

  # Apply and save diagonal scaling
  advMat_diag = advMat.getDiagonal()
  advMat_diagInv_np = np.reciprocal(np.sqrt(np.abs(advMat_diag.array)))
  advMat_diagInv = advMat.createVecLeft()
  ownRange = advMat_diagInv.owner_range
  advMat_diagInv.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), advMat_diagInv_np)
  advMat_diagInv.assemblyBegin()
  advMat_diagInv.assemblyEnd()
  advMat.diagonalScale(advMat_diagInv, None)
  curMat.diagonalScale(advMat_diagInv, None)
  preMat.diagonalScale(advMat_diagInv, None)

  advMat_diag.sqrtabs()
  advMat_diag.assemblyBegin()
  advMat_diag.assemblyEnd()

  return advMat, curMat, preMat, advMat_diag, advMat_diagInv

def setASMSubdomains(pc, subDList):
  """
  Sets up subdomains for the additive Schwarz preconditioner

  Inputs:
    pc       - The preconditioner to apply settings to
    subDList - List of unknowns associated with each subdomain
  """

  basisISList = [[] for _ in range(numProcs)]
  for ii in range(numProcs):
    if ii == proc:
      # basisISList[ii] = PETSc.IS().createGeneral(subDList[f"sub{ii}"], comm=comm)
      basisISList[ii] = PETSc.IS().createGeneral(subDList, comm=comm)
    else:
      basisISList[ii] = PETSc.IS().createGeneral([], comm=comm)
  pc.setASMLocalSubdomains(len(basisISList), is_sub=basisISList)

if __name__ == "__main__":  

  ## Load constants from options
  OptDB = PETSc.Options()
  nv = OptDB.getInt('nv', 0) # Number of vector unknowns
  ns = OptDB.getInt('ns', 0) # Number of scalar unknowns
  dt = OptDB.getReal('dt', 0) # Timestep length
  numT = OptDB.getInt('numT', 0) # Number of timesteps
  beta = OptDB.getReal('beta', 0) # Newmark-beta time constant
  simType = OptDB.getString('simType', "E")
  if simType == "E":
    n = nv
  elif simType == "A":
    n = nv+ns

  ## Set up parallel process
  comm = PETSc.COMM_WORLD
  numProcs = comm.getSize()
  proc = comm.getRank() # The current processor
  if proc == 0:
    print(f"Number of PETSc processes: {numProcs}.")

  ## Load data from input file
  filename = OptDB.getString('f', "")
  try:
    matData = np.load(filename, mmap_mode='r', allow_pickle=True)[f"arr_{proc}"].item()
  except:
    print("A valid filename for CSR values to be used in PETSc was not provided.")
    raise Exception
  # freqs = np.load(filename, mmap_mode='r', allow_pickle=True)["freqs"]

  ## Build each of the matrices in PETSc
  matStrList = []
  for key in matData: # Record all of the keys corresponding to matrices
    if "rowptr" in matData[key]:
      matStrList.append(key)
  vecLocSize = np.load(filename, mmap_mode='r', allow_pickle=True)["vecLocSizes"][proc] # Local sizes for the vector subspace
  if simType == "A":
    scalLocSize = np.load(filename, mmap_mode='r', allow_pickle=True)["scalLocSizes"][proc] # Local sizes for the scalar subspace

  matDict = {}
  
  for ii, matStr in enumerate(matStrList):
    matDict[matStr] = PETSc.Mat().create(comm=comm)
    matDict[matStr].setType('aij')

    # rowptr = matData[matStr]["rowptr"][f"proc{proc}"]
    # rowptr = rowptr - rowptr[0]
    # colval = matData[matStr]["colval"][f"proc{proc}"]
    # data = matData[matStr]["data"][f"proc{proc}"]
    rowptr = matData[matStr]["rowptr"]
    rowptr = rowptr - rowptr[0]
    colval = matData[matStr]["colval"]
    data = matData[matStr]["data"]

    # Determine matrix dimensions based on the string
    if matStr[1] == "1" or matStr[1] == "2":
      matDict[matStr].setSizes([[vecLocSize, nv], [vecLocSize, nv]])
    elif matStr[1] == "e": # Vector rows
      if matStr[2] == "e": # Vector columns
        matDict[matStr].setSizes([[vecLocSize, nv], [vecLocSize, nv]])
      elif matStr[2] == "n": # Scalar columns
        matDict[matStr].setSizes([[vecLocSize, nv], [scalLocSize, ns]])
    elif matStr[1] == "n": # Scalar rows
      if matStr[2] == "e": # Vector columns
        matDict[matStr].setSizes([[scalLocSize, ns], [vecLocSize, nv]])
      elif matStr[2] == "n": # Scalar columns
        matDict[matStr].setSizes([[scalLocSize, ns], [scalLocSize, ns]])

    matDict[matStr].setPreallocationCSR((rowptr, colval))
    matDict[matStr].setValuesCSR(rowptr, colval, data)
    matDict[matStr].assemblyBegin()
    matDict[matStr].assemblyEnd()	

  ## Set up the solver and preconditioner
  ksp = PETSc.KSP().create(comm=comm)
  pc = ksp.getPC()
  ksp.setFromOptions()
  pc.setFromOptions()
  rtol, atol, dtol, maxit = ksp.getTolerances()
  ksp.setConvergenceHistory(length=maxit)

  # Set up the time marching matrices
  advMat, curMat, preMat, advMat_diag, advMat_diagInv = buildTDMats(matDict, dt, beta, simType) 
  ksp.setOperators(A=advMat, P=advMat)
  pc.setUp()
  pc.setReusePreconditioner(True)
  
  if pc.getType() == "asm": # For additive Shwarz, set up subdomains with overlap
    if simType == "E":
      setASMSubdomains(pc, matData["vecBasisSubDs"])
    elif simType == "A":
      setASMSubdomains(pc, matData["totBasisSubDs"])

  if simType == "A":
    if pc.getType() == "fieldsplit": # For field-split, set up sub-solvers
      # Form and set up each of the subsolvers
      subKSPList = pc.getFieldSplitSubKSP()
      subPCList = [subKSPList[ii].getPC() for ii in range(len(subKSPList))]
      fieldStrList = ["vec", "scal"]
      for ii in range(len(subPCList)):
        if subPCList[ii].getType() == "asm":
          setASMSubdomains(subPCList[ii], matData[f"{fieldStrList[ii]}BasisSubDs"])

  if proc == 0:
    print(f"Solver type: {ksp.getType()}. Preconditioner type: {pc.getType()}.")
    if pc.getType() == "fieldsplit":
      for ii in range(len(subKSPList)):
        print(f"Field {ii} subsolver type: {subKSPList[ii].getType()}. Sub-preconditioner type: {subPCList[ii].getType()}")
    print(f"Relative Tolerance: {rtol}. Absolute Tolerance: {atol}. Number of unknowns: {n}.")

  ## Set up port voltage calculation
  numPorts = np.load(filename, mmap_mode='r', allow_pickle=True)["numPorts"]
  portVoltVecInds = [np.load(filename, mmap_mode='r', allow_pickle=True)[f"portVoltVecInds_{ii}"] for ii in range(numPorts)]
  portVoltVecVals_np = [np.load(filename, mmap_mode='r', allow_pickle=True)[f"portVoltVecVals_{ii}"] for ii in range(numPorts)]
  portISList = []
  portVoltVecVals = []
  numPortInds = []
  for ii, portInds in enumerate(portVoltVecInds):
    portISList.append(PETSc.IS().createGeneral(portInds, comm=comm))
    portVoltVecVals.append(PETSc.Vec().createSeq(len(portInds)))
    portVoltVecVals[ii].setValues(np.arange(len(portInds), dtype=np.int32), portVoltVecVals_np[ii])
    numPortInds.append(len(portInds))

  ## Set up for and start time marching
  prevX = preMat.createVecRight()
  curX = curMat.createVecRight()
  itNums = np.zeros((numT,), np.int32) # Iteration count at each frequency
  portV = np.zeros((numPorts, numT), dtype=np.float64) # Voltage at each port
  Vs = np.load(filename, mmap_mode='r', allow_pickle=True)["Vs"]
  # res = np.zeros((maxit, len(freqs)), np.float64) # Residual history at each frequency
  # reasons = np.zeros((len(freqs),), np.int32) # Reason for termination at each frequency
  for ii in range(2, numT-1):
    ## Set up the driving and solution vector
    b = advMat.createVecLeft()
    ownRange = b.owner_range
    if simType == "E":
      b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), matData["f_v"])
    elif simType == "A":  #TODO: Make sure this is being done correctly
      b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), np.append(matData["f_v"], matData["f_s"]))
      # b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), np.append(matData["f_s"][f"proc{proc}"], matData["f_v"][f"proc{proc}"]))
    b.assemblyBegin()
    b.assemblyEnd()
    if simType == "E":
      b.scale(-(Vs[ii+1] - Vs[ii-1])/(2*dt))
    elif simType == "A":
      b.scale(beta*Vs[ii+1] + (1-2*beta)*Vs[ii] + beta*Vs[ii-1])
    b.pointwiseMult(advMat_diagInv, b)

    # Add contributions from prior timesteps
    xArr = [curX, prevX]
    for jj, mat in enumerate([curMat, preMat]):
      tempVec = mat.createVecRight()
      mat.mult(xArr[jj], tempVec)
      b.axpy(1.0, tempVec)
      
    # Solve the problem at the current timestep
    advX = advMat.createVecRight()
    ksp.solve(b,advX)
    # advX.pointwiseMult(advMat_diagInv, b)
    itNums[ii] = ksp.getIterationNumber()
    if itNums[ii] == maxit:
      print(f"PETSc failed to converge at timestep {ii}")
      exit()

    # Record voltages
    for portNum in range(numPorts): # TODO: Sort out if I can speed this up by pre-constructing the scatterer
      portX = PETSc.Vec().createSeq(numPortInds[portNum])
      sct = PETSc.Scatter().create(advX, portISList[portNum], portX, None)
      sct.scatter(advX, portX) ## TODO: Add rescaling and account for reordering in A-phi
      portV[portNum, ii] = np.real(portX.dot(portVoltVecVals[portNum]))
    if proc == 0:
      print(f"Port voltages at timestep {ii}: {portV[:, ii]}")

    if ii == 2: # Save data from a particular timestep for debugging
      (sct, b0) = PETSc.Scatter().toZero(b)
      sct.scatter(b, b0)
      if proc == 0:
        np.save("sourceVec_TD.npy", b0.array)
      
    # Rotate the vectors to prep for the next timestep
    prevX = curX
    curX = advX
    
  if proc == 0:
    np.save("PETSc_result_TD.npy", portV)

