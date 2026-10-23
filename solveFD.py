import numpy as np
import scipy.sparse as spa
from scipy.linalg import orth
import time
# Parallelization modeles
from multiprocessing import Pool
from itertools import repeat 
from linalg_funcs import itSolve_par, saveCSRData
import subprocess, os

def solveFD(matStruct, subDLists, freqs, solveP, simType, numProcs):
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
      curVals, curRes = solveFD_PETSc(matStruct, freqs, solveP, simType, subDLists, numProcs)
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
    x_SSP, converged_SSP = ssp_refine(totFreqs[unsolvedFreqs], matStruct, curVals[:,validSols], simType)
    
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

def solveFD_PETSc(matStruct, freqs, solveP, simType, subDLists, numProcs):
  if "vecSubCutoffs" in subDLists:
    vecSubCutoffs, vecSubDs = subDLists["vecSubCutoffs"], subDLists["vecSubDs"]
    scalSubCutoffs, scalSubDs = subDLists["scalSubCutoffs"], subDLists["scalSubDs"]
  else:
    vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs = [[], [], [], []]

  csrFile = "csrData.npz"
  outFile = "PETSc_result.npy"
  resFile = "PETSc_residual.npy"
  saveCSRData(matStruct, freqs, numProcs, csrFile, vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs)
        
  cmd = buildPETScCmd("PETSc_solve.py", solveP, matStruct, csrFile, simType, vecSubCutoffs, numProcs, debugFlag=False)
  print(cmd)
  
  # Solve the problem in PETSc
  result = subprocess.run(cmd, capture_output=True, text=True)
  print(result.stdout)
  print(result.stderr)

  # Load solution and residual from files, delete files and return
  x = np.load(outFile)
  res = np.load(resFile)
  os.remove(csrFile)
  os.remove(outFile)
  os.remove(resFile)

  return x, res

def buildPETScCmd(simFile, solveP, matStruct, csrFile, simType, vecSubCutoffs, numProcs, debugFlag=False):
  # Basic PETSc setup
  nv = matStruct["K1"].shape[0] # Number of vector unknowns
  if "valgrind" in solveP and solveP["valgrind"]:
    cmd = ["valgrind", "--tool=memcheck", "--leak-check=full"]
  else:
    cmd = []
  cmd.extend(["mpiexec", "-n", f"{numProcs}", "python", f"{simFile}", "-nv", f"{nv}", "-f", csrFile, '-simType', simType, '-nsub', f"{len(vecSubCutoffs)}", '-log_view'])
  
  if debugFlag:
    cmd.extend(['-memory_view', '-mem_debug', '-ksp_view'])
  # cmd.extend(['-memory_view', '-ksp_view'])
  if simType == "A":
    ns = matStruct["Gnn"].shape[0] # Number of scalar unknowns
    cmd.extend(["-ns", f"{ns}"])

  # Solver setup
  if solveP["ksp"].lower() == "lu" or solveP["pc"].lower() == "lu": # Special setup is needed for the direct solver
    # cmd.extend(["-ksp_type", "preonly", "-pc_type", "lu", "-pc_factor_mat_ordering_type", "nd", \
    #             "-pc_factor_mat_solver_type", "mumps", "-mat_mumps_icntl_4", "1"]) # Sequential nested dissection
    if "lu-pkg" in solveP and solveP["lu-pkg"].lower() == "PETSc":
      cmd.extend(["-ksp_type", "preonly", "-pc_type", "lu", "-pc_factor_mat_ordering_type", "nd", \
                  "-pc_factor_shift_type", "nonzero"]) # Parallel ordering via ParMETIS
    else:
      cmd.extend(["-ksp_type", "preonly", "-pc_type", "lu", "-pc_factor_mat_solver_type", "mumps", \
                  "-mat_mumps_icntl_4", "1", "-mat_mumps_icntl_28", "2", "-mat_mumps_icntl_29", "2", \
                  "-mat_mumps_cntl_1", "0.1"]) # Parallel ordering via ParMETIS
      cmd.extend(["-mat_mumps_icntl_14", "500"])

  else:
    maxit, tol = solveP["maxit"], solveP["tol"]

    # Solver setup
    cmd.extend(["-ksp_type", solveP["ksp"].lower(), "-ksp_rtol", f"{tol}", "-ksp_max_it", f"{maxit}"])
    # cmd.extend(["-ksp_view", "-ksp_error_if_not_converged"])
    if solveP["ksp"].lower() == "gmres": # For GMRES, set restart tolerance and ensure breakdown tolerance isn't too low
      restart = solveP["restart"] 
      cmd.extend(["-ksp_gmres_restart", f"{restart}", "-ksp_gmres_breakdown_tolerance", f"{1.0e2}"]) 
    elif solveP["ksp"].lower() == "minres": # For MINRES, adjust settings to avoid termination due to large guess norms
      cmd.extend(["-ksp_minres_nutol", f"{1e-50}", "-ksp_minres_radius", f"{1e50}"])

      
    # Preconditioner setup
    def setPreconOptions(cmd, pc, subStr): # Unified format for modifying preconditioner options across subsolvers
      if pc.lower() == "bjacobi" or pc.lower() == "asm" or pc.lower() == "gasm": # Set up subsolvers for block decompositions

        cmd.extend([f"-{subStr}sub_ksp_type", "preonly", f"-{subStr}sub_pc_type", "lu", \
                    f"-{subStr}sub_pc_factor_mat_solver_type", "mumps", f"-{subStr}sub_mat_mumps_icntl_4", "1",
                    f"-{subStr}sub_mat_mumps_icntl_28", "2", f"-{subStr}sub_mat_mumps_icntl_29", "2", \
                    f"-{subStr}sub_mat_mumps_cntl_1", "0.1"]) # Parallel ordering via ParMETIS
        if simType == "A":
          cmd.extend([f"-{subStr}sub_mat_mumps_icntl_14", "200"])
        if debugFlag:
          cmd.extend([f"-{subStr}sub_ksp_view"])

        # if simType == "E":
        #   cmd.extend([f"-{subStr}sub_ksp_type", "preonly", f"-{subStr}sub_pc_type", "lu", \
        #               f"-{subStr}sub_pc_factor_mat_solver_type", "mumps", f"-{subStr}sub_mat_mumps_icntl_4", "1",
        #               f"-{subStr}sub_mat_mumps_icntl_28", "2", f"-{subStr}sub_mat_mumps_icntl_29", "2", \
        #               f"-{subStr}sub_mat_mumps_cntl_1", "0.1"]) # Parallel ordering via ParMETIS
        # elif simType == "A":
        #   cmd.extend([f"-{subStr}sub_ksp_type", "preonly", f"-{subStr}sub_pc_type", "lu", \
        #             f"-{subStr}sub_pc_factor_mat_ordering_type", "nd", f"-{subStr}sub_pc_factor_shift_type", "nonzero"])
        # if debugFlag:
        #   cmd.extend([f"-{subStr}sub_ksp_view"])
          
        cmd.extend([f"-{subStr}sub_ksp_error_if_not_converged"])
        if pc.lower() == "bjacobi":
          cmd.extend([f"-{subStr}pc_bjacobi_local_blocks", f"{1}"])
        elif pc.lower() == "asm":
          cmd.extend([f"-{subStr}pc_asm_type", "restrict", f"-{subStr}pc_asm_local_type", "additive"])
        elif pc.lower() == "gasm":
          cmd.extend([f"-{subStr}pc_gasm_type", "restrict", f"-{subStr}pc_gasm_view_subdomains"])
      
      elif pc.lower() == "lu-blr": # Block low-rank LU decomposition - Doesn't work very well
        eps = 1e-6 #solveP["BLR_tol"]
        cmd.extend([f"{subStr}pc_factor_mat_solver_type", "mumps", f"-{subStr}mat_mumps_icntl_4", "3", \
                    f"-{subStr}mat_mumps_icntl_28", "2", f"-{subStr}mat_mumps_icntl_29", "2", \
                    f"-{subStr}mat_mumps_icntl_35", "2", f"-{subStr}mat_mumps_cntl_7", f"{eps}", \
                    f"-{subStr}mat_mumps_icntl_36", "1"])
        
      # elif pc.lower() == "ilu": # This doesn't work because ILU isn't supported in SUPERLU_DIST, only in sequential SUPERLU
      #   cmd.extend([f"-{subStr}pc_factor_mat_solver_type", "superlu_dist", \
      #               f"-{subStr}mat_superlu_ilu_droptol", f"1e-8", \
      #               f"-{subStr}mat_superlu_ilu_fillfactor", f"600"])

      return cmd

    if solveP["pc"].lower() != "lu-blr":
      cmd.extend(["-pc_type", solveP["pc"].lower()])
    else:
      cmd.extend(["-pc_type", "lu"])
    cmd = setPreconOptions(cmd, solveP["pc"], "")
    # if simType == "E":
    #   cmd.extend(["-pc_type", solveP["pc"].lower()])
    #   cmd = setPreconOptions(cmd, solveP["pc"], "")
    # elif simType == "A":
    #   cmd.extend(["-pc_type", "fieldsplit", "-pc_fieldsplit_type", "schur", "-pc_fieldsplit_schur_fact_type", "full", "-pc_fieldsplit_schur_precondition", "a11"])
    #   cmd.extend(["-pc_type", "fieldsplit", "-pc_fieldsplit_type", "multiplicative"])
    #   for ii in range(2): # Set up subsolver info for field split
    #     if ii == 0: # Scalar potential
    #       cmd.extend([f"-fieldsplit_{ii}_ksp_type", "preonly", f"-fieldsplit_{ii}_pc_type", solveP["pc"].lower()])
    #       cmd = setPreconOptions(cmd, solveP["pc"], f"fieldsplit_{ii}_")
    #     elif ii == 1: # Vector potential
    #       cmd.extend([f"-fieldsplit_{ii}_ksp_type", "gmres", f"-fieldsplit_{ii}_ksp_max_it", "10", f"-fieldsplit_{ii}_ksp_gmres_breakdown_tolerance", f"{1.0e2}", f"-fieldsplit_{ii}_pc_type", solveP["pc"].lower()])
    #       cmd = setPreconOptions(cmd, solveP["pc"], f"fieldsplit_{ii}_")
    #   cmd.extend([f"-fieldsplit_{ii}_ksp_type", "preonly", f"-fieldsplit_{ii}_pc_type", "lu", f"-fieldsplit_{ii}_pc_factor_mat_ordering_type", "nd"])

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
    