import numpy as np
import sys
import petsc4py #type:ignore
petsc4py.init(sys.argv)
from petsc4py import PETSc #type:ignore
from mpi4py import MPI #type:ignore
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
    vecMat.axpy(1.0, matDict["K1"], PETSc.Mat.Structure.DIFFERENT)
    # vecMat.axpy(1.0, matDict["K1"], PETSc.Mat.Structure.SUBSET)
    vecMat.axpy(-1j*omega, matDict["Ree"], PETSc.Mat.Structure.DIFFERENT)
    vecMat.axpy(-1j*omega, matDict["Bee"], PETSc.Mat.Structure.SUBSET)
    vecMat.axpy(-omega**2, matDict["Tee"], PETSc.Mat.Structure.DIFFERENT)
  else:
    vecMat = matDict["K1"].copy()
  vecMat.axpy(1.0, matDict["See"], PETSc.Mat.Structure.DIFFERENT)
  if simType == "E": ## TODO: For A-phi, remember to set preconditioner re-use if using static
    A = vecMat
    vecIS, scalIS = [[], []]
  elif simType == "A":
    ep0, mu0 = [1e-9 / (36 * np.pi), 4e-7 * np.pi]
    vecMat.scale(mu0)

    scalMat = matDict["Gnn"].copy()
    scalMat.scale(-omega**2)
    if not staticFlag:
      scalMat.axpy(-1j*omega, matDict["Bnn"], PETSc.Mat.Structure.DIFFERENT)

    vecScalMat = matDict["Ken"].copy()
    vecScalMat.scale(-1j*omega)
    if not staticFlag:
      vecScalMat.axpy(-1j*omega, matDict["Ten"], PETSc.Mat.Structure.DIFFERENT)
      vecScalMat.axpy(1, matDict["Ren"], PETSc.Mat.Structure.DIFFERENT)
      vecScalMat.axpy(-1/(1j*omega), matDict["Sen"], PETSc.Mat.Structure.DIFFERENT)
    vecScalMat.scale(1/ep0)

    scalVecMat = matDict["Kne"].copy()
    scalVecMat.scale(-1j*omega*mu0*ep0)

    A = PETSc.Mat().createNest([[scalMat, scalVecMat], [vecScalMat, vecMat]], comm=comm)
    [[scalIS, _], [_, vecIS]] = A.getNestISs()
    A.convert("aij")

  ## Apply diagonal scaling
  if not staticFlag:
    A_diag = A.getDiagonal()
    A_diagInv_np = np.reciprocal(np.sqrt(np.abs(A_diag.array)))
    A_diag.destroy()
    A_diagInv = A.createVecLeft()
    ownRange = A_diagInv.owner_range
    A_diagInv.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), A_diagInv_np)
    A_diagInv.assemblyBegin()
    A_diagInv.assemblyEnd()
    A.diagonalScale(A_diagInv, A_diagInv)
  else:
    A_diagInv = []

  return A, A_diagInv, vecIS, scalIS

def setASMSubdomains(pc, totSubDList):
  """
  Sets up subdomains for the additive Schwarz preconditioner

  Inputs:
    pc       - The preconditioner to apply settings to
    totSubDList - Total list of unknowns associated with the current subdomain
  """

  # basisISList = [[] for _ in range(numProcs)]
  # for ii in range(numProcs):
  #   if ii == proc:
  #     # basisISList[ii] = PETSc.IS().createGeneral(subDList[f"sub{ii}"], comm=comm)
  #     basisISList[ii] = PETSc.IS().createGeneral(subDList, comm=comm)
  #   else:
  #     basisISList[ii] = PETSc.IS().createGeneral([], comm=comm)
  # pc.setASMLocalSubdomains(len(basisISList), is_sub=basisISList)

  # totSubDList = subDList[0]
  # for ii in range(1, len(subDList)):
  #   totSubDList = np.append(totSubDList, subDList[ii])
  basisISList = [PETSc.IS().createGeneral(totSubDList, comm=comm)]
  pc.setASMLocalSubdomains(1, is_sub=basisISList)

def setGASMSubdomains(pc, numSubDs, curSubD, subDList, subDList_ov):
  """
  Sets up subdomains for the additive Schwarz preconditioner

  Inputs:
    pc          - The preconditioner to apply settings to
    numSubDs    - Number of subdomains
    curSubD     - The subdomain (partially) owned by the current processor
    subDList    - List of unknowns associated with the current subdomain
    subDList_ov - List of unknowns in the current subdomain, including overlap
  """

  # basisISList = [[] for _ in range(numSubDs)]
  # basisISList_ov = [[] for _ in range(numSubDs)]
  # for ii in range(numSubDs):
  #   if ii == curSubD:
  #     # basisISList[ii] = PETSc.IS().createGeneral(subDList[f"sub{ii}"], comm=comm)
  #     basisISList[ii] = PETSc.IS().createGeneral(subDList, comm=comm)
  #     basisISList_ov[ii] = PETSc.IS().createGeneral(subDList_ov, comm=comm)
  #   else:
  #     basisISList[ii] = PETSc.IS().createGeneral([], comm=comm)
  #     basisISList_ov[ii] = PETSc.IS().createGeneral([], comm=comm)
  # pc.GASMSetSubdomains(numSubDs, iis=basisISList, ois=basisISList_ov)

  # basisISList = [PETSc.IS().createGeneral(subDList, comm=comm)]
  # basisISList_ov = [PETSc.IS().createGeneral(subDList_ov, comm=comm)]
  # pc.GASMSetSubdomains(1, iis=basisISList, ois=basisISList_ov)

  # basisISList = []
  # basisISList_ov = []
  # print(f"subDList_ov for Processor {proc}: {subDList_ov}")
  # for ii in range(numSubDs): # Check whether this processor has unknowns from each subdomain
  #   if ii == curSubD: # For its own subdomain, add the local info
  #     print(f"subDList for Processor {proc} (subdomain {ii}): {subDList}")
  #     basisISList.append(PETSc.IS().createGeneral(subDList, comm=comm))
  #     basisISList_ov.append(PETSc.IS().createGeneral(subDList, comm=comm))
  #   elif len(subDList_ov[ii]) != 0: # This subdomain contains unknowns from subdomain ii
  #     print(f"Got here on processor {proc}")
  #     basisISList.append(PETSc.IS().createGeneral([], comm=comm))
  #     basisISList_ov.append(PETSc.IS().createGeneral(subDList_ov[ii], comm=comm))
  #     print(f"Processor {proc} outer list for subdomain {ii}: {basisISList_ov.getIndices()}")
  # print(f"Number of subdomains for processor {proc}: {len(basisISList)}")
  # pc.GASMSetSubdomains(len(basisISList), iis=basisISList, ois=basisISList_ov)

  basisISList = []
  basisISList_ov = []
  subDList_ov[curSubD] = subDList
  subDList_noOv = [[] for _ in range(numSubDs)]
  subDList_noOv[curSubD] = subDList
  for ii in range(numSubDs):
    if len(subDList_ov[ii]) != 0: # If the subdomain isn't empty, mark it as so
      color = 1
    else:
      color = MPI.UNDEFINED
    # print(f"Color on processor {proc} for subdomain {ii}: {color}")
    subcomm = PETSc.COMM_WORLD.tompi4py().Split(color, proc)
    if color == 1:
      basisISList.append(PETSc.IS().createGeneral(subDList_noOv[ii], comm=subcomm))
      basisISList_ov.append(PETSc.IS().createGeneral(subDList_ov[ii], comm=subcomm))
      # print(f"Processor {proc} intercommunicator check: {subcomm.Is_inter()}")
      # print(f"Processor {proc} intracommunicator check: {subcomm.Is_intra()}")
      subcomm.Free()
      # basisISList[-1].view() 
    
  # basisISList = [basisISList[activeSubDs[ii]] for ii in range(len(activeSubDs))]
  # basisISList_ov = [basisISList_ov[activeSubDs[ii]] for ii in range(len(activeSubDs))]
  # print(f"Subdomain list length on processor {proc}: {len(basisISList)}")
  pc.GASMSetSubdomains(len(basisISList), iis=basisISList, ois=basisISList_ov)
  # for ii in range(len(basisISList)):
  #   basisISList[ii].destroy()
  #   basisISList_ov[ii].destroy()
  # pc.GASMSetSubdomains(numSubDs, iis=basisISList, ois=basisISList_ov)

if __name__ == "__main__":  

  ## Load constants from options
  OptDB = PETSc.Options()
  nv = OptDB.getInt('nv', 0) # Number of vector unknowns
  ns = OptDB.getInt('ns', 0) # Number of scalar unknowns
  simType = OptDB.getString('simType', "E")
  outFile = OptDB.getString('outFileZ',"PETSc_resultZ.npy")
  resFile = OptDB.getString('resFileZ',"PETSc_residualZ.npy")
  if simType == "E":
    n = nv
  elif simType == "A":
    n = nv+ns
  numSubDs = OptDB.getInt('nsub', 0) # Number of subdomains
  mem_debug = OptDB.hasName('mem_debug')

  ## Set up parallel process
  comm = PETSc.COMM_WORLD
  numProcs = comm.getSize()
  proc = comm.getRank() # The current processor
  if proc == 0:
    print(f"Number of PETSc processes: {numProcs}.")
    # print(f"MPI version: {MPI.Get_library_version()}")

  ## Load data from input file
  filename = OptDB.getString('f', "")
  try:
    matData = np.load(filename, mmap_mode='r', allow_pickle=True)[f"arr_{proc}"].item()
  except:
    print("A valid filename for CSR values to be used in PETSc was not provided.")
    raise Exception
  freqs = np.load(filename, mmap_mode='r', allow_pickle=True)["freqs"]

  ## Build each of the matrices in PETSc
  matStrList = []
  for key in matData: # Record all of the keys corresponding to matrices
    if isinstance(matData[key], dict) and "rowptr" in matData[key]:
      matStrList.append(key)
  vecLocSize = np.load(filename, mmap_mode='r', allow_pickle=True)["vecLocSizes"][proc] # Local sizes for the vector subspace
  if simType == "A":
    scalLocSize = np.load(filename, mmap_mode='r', allow_pickle=True)["scalLocSizes"][proc] # Local sizes for the scalar subspace

  matDict = {}
  
  for ii, matStr in enumerate(matStrList):
    matDict[matStr] = PETSc.Mat().create(comm=comm)
    matDict[matStr].setType('aij')

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
  ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED) # Doesn't work with MINRES

  if simType == "A":
    P, _, _, _ = buildFDMat(matDict, freqs[0], simType, staticFlag=True) 
    if pc.getType() == "fieldsplit": # For field-split, set up sub-solvers
      # Build the static preconditioner and set up the preconditioner
      # Even if static preconditioning isn't used, this is necessary to set up subdomains for ASM
      ksp.setOperators(A=P, P=P)
      pc.setUp()
    #   pc.setReusePreconditioner(True) # Uncomment to use static preconditioner

      # # Form and set up each of the subsolvers
      subKSPList = pc.getFieldSplitSubKSP()
      subPCList = [subKSPList[ii].getPC() for ii in range(len(subKSPList))]
      fieldStrList = ["scal", "vec"]
      for ii in range(len(subPCList)):
        if subPCList[ii].getType() == "asm":
          setASMSubdomains(subPCList[ii], matData[f"{fieldStrList[ii]}BasisSubDs"])
          


  if proc == 0:
    print(f"Solver type: {ksp.getType()}. Preconditioner type: {pc.getType()}.")
    if pc.getType() == "fieldsplit":
      for ii in range(len(subKSPList)):
        print(f"Field {ii} subsolver type: {subKSPList[ii].getType()}. Sub-preconditioner type: {subPCList[ii].getType()}")
    print(f"Relative Tolerance: {rtol}. Absolute Tolerance: {atol}. Number of unknowns: {n}.")

  ## Set up and solve the problem at each frequency
  xArr = np.zeros((n, len(freqs)), np.complex128)
  itNums = np.zeros((len(freqs),), np.int32) # Iteration count at each frequency
  res = np.zeros((maxit, len(freqs)), np.float64) # Residual history at each frequency
  reasons = np.zeros((len(freqs),), np.int32) # Reason for termination at each frequency
  for ii, freq in enumerate(freqs):
    omega = 2*np.pi*freq

    A, A_diagInv, vecIS, scalIS = buildFDMat(matDict, freq, simType)
    symFlag = A.isSymmetric()
    if proc == 0:
      print(f"Matrix symmetry flag: {symFlag}")

    # print(f"Processor {proc} scalar indices: {scalIS.getIndices()}")

    ## Set up the driving and solution vector
    b = A.createVecLeft()
    ownRange = b.owner_range
    if simType == "E":
      b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), matData["f_v"])
    elif simType == "A":  #TODO: Make sure this is being done correctly
      b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), np.append(matData["f_s"], matData["f_v"]))
      # b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), np.append(matData["f_s"][f"proc{proc}"], matData["f_v"][f"proc{proc}"]))
    b.assemblyBegin()
    b.assemblyEnd()
    if simType == "E":
      b.scale(1j*2*np.pi*freq)
    b.pointwiseMult(A_diagInv, b)
      
    x = A.createVecRight()
    if simType == "E":
      ksp.setOperators(A=A, P=A)
    elif simType == "A":
      P.diagonalScale(A_diagInv, A_diagInv) # TODO: Make sure this works correctly for multiple frequencies
      ksp.setOperators(A=A, P=A)
    ksp.setTolerances(rtol=rtol, atol=atol, divtol=dtol, max_it=maxit)
  
    ## Solve the problem
    start_time = time.time()
    ksp.solve(b,x)
  
    # res = A*x - b
    itNums[ii] = ksp.getIterationNumber()
    if itNums[ii] != 1:
      curRes = ksp.getConvergenceHistory()
      res[:len(curRes),ii] = curRes
    reasons[ii] = ksp.getConvergedReason()

    if mem_debug: # DEBUGGING for memory issues
      if pc.getType() == "lu":
        lu = pc.getFactorMatrix()
        memUsage = lu.getMumpsInfo(22)
        nzCount = lu.getMumpsInfo(24)
        print(f"Memory usage from LU factorization on processor {proc}: {memUsage} MB. Number of nonzeros: {nzCount}")

        totNzCount = lu.getMumpsInfog(20)
        totMem = lu.getMumpsInfog(22)
        maxMem = lu.getMumpsInfog(21)
        if proc == 0:
          print(f"Total memory usage in the LU factorization across all processors: {totMem} MB. " \
                f"Maximum for a single processor: {maxMem} MB. Total number of nonzero entries: {totNzCount:.3e}")
        lu.destroy()
      elif pc.getType() == "asm" or pc.getType() == "gasm":
        memUsage, nzCount = [0, 0]
        if pc.getType() == "asm": 
          subKSPList = pc.getASMSubKSP()
        elif pc.getType() == "gasm":
          subKSPList = pc.GASMGetSubKSP()
        for jj, subKSP in enumerate(subKSPList):
          subPC = subKSP.getPC()
          subLU = subPC.getFactorMatrix()
          memUsage += subLU.getMumpsInfo(22)
          nzCount += subLU.getMumpsInfo(24)
          subLU.destroy()
          # totNzCount += subLU.getMumpsInfog(20)
          # totMem += subLU.getMumpsInfog(22)
          # maxMem += subLU.getMumpsInfog(21)
        
          # print(f"Memory usage for the {jj}th subdomain on processor {proc}: {memUsage} MB.")
        print(f"Total memory usage from subdomain LU factorizations on processor {proc}: {memUsage} MB. Number of nonzeros: {nzCount}")
        
        # Combine info from each subdomain
        totNzVec = PETSc.Vec().createMPI(numProcs)
        totNzVec.setValue(proc, nzCount)
        (sct, totNzVec0) = PETSc.Scatter().toZero(totNzVec) 
        sct.scatter(totNzVec, totNzVec0)
        sct.destroy()
        totMemVec = PETSc.Vec().createMPI(numProcs)
        totMemVec.setValue(proc, memUsage)
        (sct, totMemVec0) = PETSc.Scatter().toZero(totMemVec) 
        sct.scatter(totMemVec, totMemVec0)
        sct.destroy()
        if proc == 0:
          print(f"Total memory usage in the LU factorization across all processors and subdomains: {np.sum(totMemVec0.array)} MB. "\
                f"Maximum for a single processor: {np.max(totMemVec0.array)} MB. Total number of nonzero entries: {np.sum(totNzVec0.array):.3e}")
        totNzVec.destroy()
        totNzVec0.destroy()
        totMemVec.destroy()
        totMemVec0.destroy()
  
    ## Rescale and save the global solution
    # x.pointwiseMult(A_diagInv, x)
    if simType == "E":
      (sct, vecX0) = PETSc.Scatter().toZero(x)
      sct.scatter(x, vecX0)
    elif simType == "A":
      
      # [[scalIS, _], [_, vecIS]] = A.getNestISs()

      # Move vector unknown data into its own vector
      vecX = matDict["K1"].createVecLeft()
      sct = PETSc.Scatter().create(x, vecIS, vecX, None)
      sct.scatter(x, vecX)
      # Move vector data to processor 0
      (sct, vecX0) = PETSc.Scatter().toZero(vecX) 
      sct.scatter(vecX, vecX0)

      # Move scalar data into its own vector
      scalX = matDict["Gnn"].createVecLeft()
      sct = PETSc.Scatter().create(x, scalIS, scalX, None)
      sct.scatter(x, scalX)
      # Move scalar data to processor 0
      (sct, scalX0) = PETSc.Scatter().toZero(scalX) 
      sct.scatter(scalX, scalX0)



    bNorm = b.norm(PETSc.NormType.NORM_2)
    resVec = A.createVecRight()
    b.scale(-1)
    A.multAdd(x, b, resVec) # Compute the final residual vector
    resFinal = resVec.norm(PETSc.NormType.NORM_2)/bNorm
    if proc == 0:
      xArr[:nv,ii] = vecX0.array
      if simType == "A": # Move scalar data into xArr
        xArr[nv:,ii] = scalX0.array
      res[:,ii] /= bNorm
      print(f"PETSc solve time for {freq/1e9:.3f} GHz: {(time.time() - start_time):.5f} s. Convergence reason: {reasons[ii]}",flush=True)
      print(f"Final relative residual: {res[np.min([itNums[ii],maxit-1]),ii]:.3e}. Iteration count: {itNums[ii]}.",flush=True)
      print(f"Exact final relative residual: {resFinal:.3e}",flush=True)

    A.destroy()
    b.destroy()
    x.destroy()
    resVec.destroy()
    ksp.reset()
    pc.reset()
    pc.destroy()
    pc = ksp.getPC()
    
  if proc == 0:
    np.save(outFile, xArr)
    np.save(resFile, res)

