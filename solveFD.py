import numpy as np
import scipy.sparse as spa
from scipy.linalg import orth
import time
# Parallelization modeles
from multiprocessing import Pool
from itertools import repeat 
from linalg_funcs import itSolve_par, saveCSRData
import subprocess, os

def solveFD(matStruct, subDLists, freqs, solveP, simType, numProcs, res_file_name):
  """
  Evaluates a frequency domain problem given the finite element matrices and driving vector

  Inputs:
    matStruct     - Dictionary containing the matrices and driving vector
    subDLists     - Dictionary containing information about the subdomains
    freqs         - Vector of frequencies to solve the problem at
    solveP        - Dictionary containing parameters of the solver
    simType       - Type of simulation being run ("E", "A", or "TCS")
    numProcs      - Number of processors available to parallelize over
    res_file_name - File path used to write PETSc results (if needed)
  
  Outputs:
    totVals       - Total solution vector at each frequency (stored as an array)
    diagnostics   - Dictionary containing diagnostic information

  """
  # Set constants
  ep0 = 1e-9 / (36 * np.pi)
  mu0 = 4e-7 * np.pi
  K1_coeff = mu0
  Ken_coeff = 1/ep0

  if simType == "E" or simType == "A" or simType == "TCS":
    numUnknowns = matStruct["K1"].shape[0]
    if simType == "A" or simType == "TCS":
      numEdgeUnknowns = numUnknowns
      numUnknowns += matStruct["Gnn"].shape[0]
  elif simType == "Phi":
    numUnknowns = matStruct["Gnn"].shape[0]

  diagnostics = {}
  diagnostics["iter"] = np.zeros((len(freqs,)), dtype=np.int32)
  if solveP["pkg"].lower() == "petsc":
    diagnostics["res"] = np.zeros((solveP["maxit"], len(freqs)), dtype=np.float64)
  
  totFreqs = np.copy(freqs)
  numTotFreqs = np.size(totFreqs)
  #print(f"numTotFreqs is equal to {numTotFreqs }")
  unsolvedFreqs = np.ones((numTotFreqs,), dtype=bool) # Frequency indices where the solution hasn't been found
  totVals = np.zeros((numUnknowns, numTotFreqs), dtype=np.complex128)

  sspThresh = 10 # Threshold for using SSP - change to be dynamic
  if solveP["adaptSweep"] and numTotFreqs > sspThresh: 
    curFreqInds = np.floor(np.linspace(0, numTotFreqs-1, sspThresh)).astype(np.int32)
    freqs = freqs[curFreqInds]
  else:
    curFreqInds = np.arange(numTotFreqs)
  numCurFreqs = len(freqs) # Number of frequencies for the current SSP iteration
  
  while numCurFreqs > 0: # Loop through SSP passes
    if solveP["pkg"].lower() == "scipy":
      ## Set up composite matrices for either field-based or potential-based methods
      A, b, M_diag, A_static, newInds = buildSysFD(matStruct, freqs, simType)
    else:
      _, _, M_diag, _, newInds = buildSysFD(matStruct, freqs, simType)
  
    ## Solve the system
    curVals = np.zeros((matStruct["f"].size, numCurFreqs), dtype=np.complex128)
    if solveP["pkg"].lower() == "scipy":
      if solveP["preType"] == "Static" and simType != "A":
          print("Static preconditioning can only be used in A-Phi. Defaulting to a dynamic preconditioner") 
          solveP["preType"] = "Dynamic"
      if solveP["preType"] == "Dynamic":
        p = Pool(np.min([numProcs, len(freqs)]))
        res = list(p.starmap(itSolve_par, zip(A, b, freqs, repeat(solveP))))
        for ii in range(numCurFreqs):
          curVals[:,ii] = res[ii][0]
          diagnostics["iter"][curFreqInds[ii]] = res[ii][1]
      elif solveP["preType"] == "Static":
        start_time = time.time()
        options = dict(ILU_MILU='SILU', DiagPivotThresh=0.01, SymmetricMode=True)
        A_static_ilu = spa.linalg.spilu(A_static.astype(np.complex128).tocsc(), drop_tol=solveP["dropTol"], fill_factor=(solveP["rowFill"]), options=options)
        M = spa.linalg.LinearOperator(A_static.shape, A_static_ilu.solve)
        print(f"Preconditioning done. Time elapsed: {(time.time() - start_time):.3f} s." \
            f" Sparsity percentage of M: {(100*(1-(A_static_ilu.L.nnz+A_static_ilu.U.nnz)/A_static.shape[0]**2)):.3f}%")
        
        for ii in range(numCurFreqs):
          curVals[:,ii], diagnostics["iter"][curFreqInds[ii]] = itSolve_par(A[ii], b[ii], freqs[ii], solveP, M=M, debug=True)
    elif solveP["pkg"].lower() == "petsc":
      curVals, curRes = solveFD_PETSc(matStruct, freqs, solveP, simType, subDLists, numProcs, res_file_name)
      if "ksp" in solveP and (solveP["ksp"].lower() != "lu" and solveP["pc"].lower() != "lu"):
        # If the solver isn't direct, save convergence info
        for jj, freqInd in enumerate(curFreqInds):
          diagnostics["iter"][freqInd] = np.sum(curRes[:,jj] != 0)
          diagnostics["res"][:, freqInd] = curRes[:,jj]
    
    ## Store the full solutions
    validSols = np.zeros((len(curFreqInds),), dtype=np.bool_) # Mark which frequencies the solution is valid for
    for ii, ind in enumerate(curFreqInds):
      if not np.isnan(curVals[:,ii]).any() and not np.isinf(curVals[:,ii]).any(): # Make sure the solution isn't NaN or inf before marking as solved
        unsolvedFreqs[ind] = False # Mark that the problem has been solved here
        validSols[ii] = True
        totVals[newInds,ind] = M_diag[ii] @ curVals[:,ii]
        for jj, dirInds in enumerate(matStruct["knownInds"]): # Add dirichlet values back into the vector
          totVals[dirInds,ind] = matStruct["knownVals"][jj]
        if simType == "A": # Rescale by constants for A-phi
          totVals[:,ind][:numEdgeUnknowns] = totVals[:,ind][:numEdgeUnknowns]*K1_coeff
          totVals[:,ind][numEdgeUnknowns:] = totVals[:,ind][numEdgeUnknowns:]*Ken_coeff

    ## Use SSP to interpolate solutions at other frequecies
    x_SSP, converged_SSP = ssp_refine(totFreqs[unsolvedFreqs], matStruct, curVals[:,validSols], simType,1e-1)
    
    ## Store the valid SSP solutions
    sspInds = np.arange(numTotFreqs)[unsolvedFreqs]
    for ii, ind in enumerate(sspInds):
      if converged_SSP[ii]:
        unsolvedFreqs[ind] = False # Mark that the problem has been solved here
        totVals[newInds,ind] = x_SSP[:,ii]
        for jj, dirInds in enumerate(matStruct["knownInds"]): # Add dirichlet values back into the vector
          totVals[dirInds,ind] = matStruct["knownVals"][jj]
        if simType == "A": # Rescale by constants for A-phi
          totVals[:,ind][:numEdgeUnknowns] = totVals[:,ind][:numEdgeUnknowns]*K1_coeff
          totVals[:,ind][numEdgeUnknowns:] = totVals[:,ind][numEdgeUnknowns:]*Ken_coeff

    ## Set up the next SSP iteration
    numUnsolvedFreqs = np.sum(unsolvedFreqs)
    unsolvedInds = np.where(unsolvedFreqs == True)[0]
    if numUnsolvedFreqs > sspThresh:
      curFreqInds = unsolvedInds[np.floor(np.linspace(0, numUnsolvedFreqs-1, sspThresh)).astype(np.int32)]
    else:
      curFreqInds = unsolvedInds[np.arange(numUnsolvedFreqs)]
    freqs = totFreqs[curFreqInds]
    numCurFreqs = len(freqs)
    
  return totVals, diagnostics

def solveFD_PETSc(matStruct, freqs, solveP, simType, subDLists, numProcs, res_file_name):
  """
  Sets up a PETSc subprocess to solve a frequency domain problem at the provided set of frequencies

  Inputs:
    matStruct     - Dictionary containing the matrices and driving vector
    freqs         - Vector of frequencies to solve the problem at
    solveP        - Dictionary containing parameters of the solver
    simType       - Type of simulation being run ("E", "A", or "TCS")
    subDLists     - Dictionary containing information about the subdomains
    numProcs      - Number of processors available to parallelize over
    res_file_name - File path used to write PETSc results
  
  Outputs:
    x    - Total solution vector at each frequency (stored as an array)
    res  - Residual vector for each solve
  """
  if "vecSubCutoffs" in subDLists:
    vecSubCutoffs, vecSubDs = subDLists["vecSubCutoffs"], subDLists["vecSubDs"]
    scalSubCutoffs, scalSubDs = subDLists["scalSubCutoffs"], subDLists["scalSubDs"]
  else:
    vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs = [[], [], [], []]
    
  basePath = "./"+os.path.splitext(os.path.basename(res_file_name))[0]
  csrFile = basePath+"csrData.npz"
  outFile = basePath+"PETSc_result.npy"
  resFile = basePath+"PETSc_residual.npy"
  saveCSRData(matStruct, freqs, numProcs, csrFile, vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs)
        
  cmd = buildPETScCmd("PETSc_solve.py", solveP, matStruct, csrFile, outFile, resFile, simType, vecSubCutoffs, numProcs, debugFlag=False)
  # print(cmd)
  
  # Solve the problem in PETSc
  result = subprocess.run(cmd, capture_output=True, text=True)
  # print(result.stdout)
  # print(result.stderr)

  # Load solution and residual from files, delete files and return
  x = np.load(outFile)
  res = np.load(resFile)
  os.remove(csrFile)
  os.remove(outFile)
  os.remove(resFile)

  return x, res

def buildPETScCmd(simFile, solveP, matStruct, csrFile, outFile, resFile, \
                  simType, vecSubCutoffs, numProcs, debugFlag=False):
  """
  Constructs a command string to run a PETSc solve of the matrix

  Inputs:
    simFile       - Path to the file used to run the PETSc solution process
    solveP        - Dictionary containing parameters of the solver
    matStruct     - Dictionary containing the matrices and driving vector
    csrFile       - Path to the file containing all of the matrices in parallelized form
    outFile       - Path to the PETSc output file
    resFile       - Path to the PETSc residual file    
    simType       - Type of simulation being run ("E", "A", or "TCS")
    vecSubCutoffs - Vector of subdomain cutoffs (only needed for domain decomposition)
    numProcs      - Number of processors available to parallelize over
    debugFlag     - Indicates whether to print out additional debug information

  Outputs:
    cmd           - String providing the command used to run the PETSc solve process
  """
  nv = matStruct["K1"].shape[0] # Number of vector unknowns
  if "valgrind" in solveP and solveP["valgrind"]:
    cmd = ["valgrind", "--tool=memcheck", "--leak-check=full"]
  else:
    cmd = []
  cmd.extend(["mpiexec", "-n", f"{numProcs}", "python", f"{simFile}", "-nv", f"{nv}", "-f", csrFile, \
              '-simType', simType, '-nsub', f"{len(vecSubCutoffs)}", "-outFileZ", outFile, "-resFileZ", resFile, '-log_view'])
  
  if debugFlag:
    cmd.extend(['-memory_view', '-mem_debug', '-ksp_view'])
  if simType == "A":
    ns = matStruct["Gnn"].shape[0] # Number of scalar unknowns
    cmd.extend(["-ns", f"{ns}"])

  # Set direct solver settings
  cmd.extend(["-ksp_type", "preonly", "-pc_type", "lu", "-pc_factor_mat_solver_type", "mumps", \
              "-mat_mumps_icntl_4", "1", "-mat_mumps_icntl_28", "2", "-mat_mumps_icntl_29", "2", \
              "-mat_mumps_cntl_1", "0.1"]) # Parallel ordering via ParMETIS
  if solveP["singFlag"]:
    # If singular functions are used, increase the memory allocation for the factorization
    cmd.extend(["-mat_mumps_icntl_14", "500"])


  return cmd

def buildSysFD(matStruct, freqs, simType):
  """
  Builds the system of equations to solve for frequency domain problems

  Inputs:
    matStruct - Dictionary containing the individual FEM matrices for each operator
    freqs     - Vector of frequencies to construct systems for
    simType   - The type of simulation the system is being built for ("E", "A", "TCS")

  Outputs:
    A         - List of matrices for each frequency
    b         - List of source vectors for each frequency
    M_diag    - List of diagonal preconditioners for each frequency
    A_static  - Static (frequency independent) version of the matrix for preconditioning (only for "A")
  """
  if simType == "E" or simType == "A" or simType == "TCS":
    K1 = matStruct["K1"]
    K2 = matStruct["K2"]
    Bee = matStruct["Bee"]
    Ree = matStruct["Ree"]
    Tee = matStruct["Tee"]
    See = matStruct["See"]
    f = matStruct["f"]
    if simType == "A" or simType == "TCS":
      Ken = matStruct["Ken"]
      Kne = matStruct["Kne"]
      Gnn = matStruct["Gnn"]
      Ren = matStruct["Ren"]
      Ten = matStruct["Ten"]
      Sen = matStruct["Sen"]
      Bnn = matStruct["Bnn"]
      if simType == "TCS":
        Rne = matStruct["Rne"]
        Rnn = matStruct["Rnn"]
        Tne = matStruct["Tne"]
        Sne = matStruct["Sne"]
        Tnn = matStruct["Tnn"]
        Snn = matStruct["Snn"]
        Ben = matStruct["Ben"]
        Bne = matStruct["Bne"]
  elif simType == "Phi":
    Gnn = matStruct["Gnn"]
    Tnn = matStruct["Tnn"]

  ep0 = 1e-9 / (36 * np.pi)
  mu0 = 4e-7 * np.pi

  K1_coeff = mu0
  Ken_coeff = 1/ep0
  Kne_coeff = mu0*ep0
  Gnn_coeff = 1

  A, b, M_diag, A_static = [[],[],[],[]] # Set the list of matrices to initially be empty
  if simType == "A":
    omega = 2*np.pi*freqs[0]
    A_static = spa.block_array([[K1_coeff*(K1), -Ken_coeff*(1j*omega*Ken)], 
                                [-Kne_coeff*1j*omega*Kne, Gnn_coeff*(-omega**2*Gnn)]])

  for freqVal in freqs:
    omega = 2*np.pi*freqVal
    if simType == "E":
      A.append(K1+See - (omega**2)*(K2+Tee) - 1j*omega*(Ree+Bee))
      b.append(1j*omega*f)
    elif simType == "A":
      A.append(spa.block_array([[K1_coeff*((K1+See) - omega**2*(K2+Tee) - 1j*omega*(Ree+Bee)), -Ken_coeff*(1j*omega*(Ken+Ten) - Ren + Sen/(1j*omega))], 
                                [-Kne_coeff*1j*omega*Kne, Gnn_coeff*(-omega**2*Gnn - 1j*omega*Bnn)]]))#[perm[:,None], perm])
      b.append(f)
    elif simType == "TCS":
      A.append(spa.block_array([[(K1 - omega**2*K2 - 1j*omega*(Ree+Bee)), -omega**2*Ken - 1j*omega*(Ren+Ben)], 
                                [-omega**2*Kne - 1j*omega*(Rne+Bne), -omega**2*Tnn - 1j*omega*(Rnn+Bnn)]]))
      b.append(1j*omega*f)
    elif simType == "Phi":
      A.append(Tnn)
      b.append(np.zeros((Tnn.shape[0],), dtype=np.float64))
    M_diag.append(spa.diags_array(1/np.sqrt(np.abs(A[-1].diagonal())), format='csr')) # Form and apply a diagonal preconditioner
    A[-1] = M_diag[-1] @ A[-1] @ M_diag[-1]
    b[-1] = M_diag[-1] @ b[-1]

  ## If a dirichlet boundary exists, account for it
  if len(matStruct["knownVals"]) != 0: 
    A, b, newInds = dirichletBoundary(A, b, M_diag, matStruct["knownVals"], matStruct["knownInds"])
    for ii in range(len(M_diag)):
      M_diag[ii] = M_diag[ii][:,newInds][newInds,:]
    if A_static != []:
      A_static = A_static[:,newInds][newInds,:]
  else:
    newInds = np.arange(len(b[0]))

  return A, b, M_diag, A_static, newInds

def ssp_refine(freqs, matStruct, sols, simType, threshold=1e-4):
  """
  function for testing the current SSP solution at specified frequencies, and returns
  a list of the points at which the error is above tolerance

  Inputs
  freqs - list of test frequency points
  A -  list of the matrices
  b -  list of the source vectors
  sols - list of the solution vectors so far that generate the solution space.
  """

  # form orthogonal matrix out of the solution vectors to create projection matrix XH
  X = orth(sols)
  XH = X.conj().T

  # solve reduced system for each test frequency to check residual
  vals = np.zeros((len(sols), len(freqs)), dtype=sols.dtype) # Solution vector at each point
  convergedPoints = np.zeros((len(freqs),), dtype=bool) # frequency points where SSP converges
  for ii in range(len(freqs)):

    A, b, M_diag, _, _ = buildSysFD(matStruct, np.array([freqs[ii]]), simType)
    A = A[0]
    b = b[0]
    M_diag = M_diag[0]

    sim = (XH @ (A @ X))   # congruent matrix of A projected on solution space of X
    z = (XH @ b)    # projection of source vector onto SS

    v = np.linalg.solve(sim, z)     # reduced solution
    x = X @ v    # transformation of reduced solution back
    vals[:,ii] = M_diag @ x

    # get residual
    res = np.linalg.norm(b - (A @ (x)).T) / np.linalg.norm(b)
    # print(f"Residual at {freqs[ii]/1e9:.3f} GHz: {res:.3e}")
      
    if res <= threshold:
      convergedPoints[ii] = True

  return vals, convergedPoints

def dirichletBoundary(A, b, M_diag, knownVals, knownInds):
  """
  Applies a Dirichlet boundary condition to the system

  Inputs:
    A         - The original FEM matrix (including the Dirichlet boundary unknowns)
    b         - The original source vector (including the Dirichlet boundary unknowns)
    knownVals - Values of each set of unknowns on the Dirichlet boundary
    knownInds - Indices of the values in knownVals
  """

  # Find the set of new indices in the matrices
  removedInds = knownInds.flatten()
  newInds = np.setdiff1d(np.arange(len(b[0])), removedInds)

  # Update the source term for each set of known values and reduce the dimension of the matrix
  for freqPt in range(len(A)):
    for ii, vals in enumerate(knownVals):
      for jj, val in enumerate(vals):
        curInd = knownInds[ii,jj]
        b[freqPt] -= (A[freqPt][:,[curInd]] * (val/M_diag[freqPt][curInd, curInd])).todense().flatten()
    b[freqPt] = b[freqPt][newInds]
    A[freqPt] = A[freqPt][:,newInds][newInds,:]

  return A, b, newInds
    
