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
import scipy.sparse as spa
import time
import subprocess, os # For working with PETSc

def itSolve_par(A, b, freq, solveP, atol=0, M=None, x0=None, debug=False, saveDiagnostics=True):
  '''
  Function to enable use of the iterative solver in parallel to solve Ax = b
    Inputs:
      A       - left-hand side matrix
      b       - right-hand side source vector
      freq    - frequency the problem is being evaluated at
      solveP  - dictionary of solver parameters
      M       - preconditioner matrix
      x0      - initial guess at the solution
      debug   - a flag to indicate whether to print debugging statements to the terminal
    Outputs:
      x       - solution vector for Ax = b
  '''

  if solveP["ksp"].lower() == "lu" or solveP["pc"].lower() == "lu":
    # Solve the problem using a direct solver
    A_lu = spa.linalg.splu(A.tocsc())
    x = [A_lu.solve(b)]
    counter=itSolve_counter()
    counter.niter = 0
    perm_inv = np.arange(len(b))

  else:
    # Run an iterative solution process
    rtol, restart, maxit = [solveP["tol"], solveP["restart"], solveP["maxit"]] # Solver settings
    dropTol, rowFill = [solveP["dropTol"], solveP["rowFill"]]
    
    # Instantiate a counter object for tracking relative residual and iteration count of solver
    counter=itSolve_counter()

    # If a preconditioner isn't provided, perform ILU preconditioning
    if M == None:
      # First permute the matrix to speed up preconditioning
      perm = spa.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
      perm_inv = np.argsort(perm)
      A = A[perm[:,None], perm]
      b = b[perm]

      start_time = time.time()
      options = dict(ILU_MILU='SILU', DiagPivotThresh=0.0)
      # A_ilu = spa.linalg.spilu(A.tocsc(), drop_tol=dropTol, fill_factor=((fillPct/100)*A.shape[0]**2/A.nnz), options=options)
      A_ilu = spa.linalg.spilu(A.tocsc(), drop_tol=dropTol, fill_factor=rowFill, options=options)
      M = spa.linalg.LinearOperator(A.shape, A_ilu.solve)
      if debug:
        print(f"Preconditioning done for {freq:.3e} Hz. Time elapsed: {(time.time() - start_time):.3f} s." \
              f" Sparsity percentage of M: {(100*(1 - (A_ilu.L.nnz+A_ilu.U.nnz)/A.shape[0]**2)):.3f}%")
    else:
      perm = np.arange(A.shape[0])
      perm_inv = perm

    # Run iterative solver
    start_time = time.time()
    x = spa.linalg.gmres(A, b, restart=restart, maxiter=maxit, x0=x0,
                        rtol=rtol, atol=atol, M=M, callback=counter, callback_type='pr_norm')
    relres = np.linalg.norm(A.dot(x[0]) - b) / np.linalg.norm(b)
    if x[1] == 0: # Indicates whether the solver converged
      if debug:
        try:
          print(f"Iterative solver converged for {freq:.3e} Hz. Time elapsed: {(time.time() - start_time):.3f} s." \
              f" Iteration count: {counter.niter}. Relative residual: {relres:.3e}")
        except:
          print("converged")
    else:
      print(f"Iterative solver did not converge for {freq:.3e} Hz after {maxit} iterations. Time elapsed:  {(time.time() - start_time):.3f} s. Final relative residual: {relres:.3e}")
  if saveDiagnostics:
    return x[0][perm_inv], counter.niter
  else:
    return x[0][perm_inv]

# From https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
class itSolve_counter(object):
    def __init__(self):
        self.relres = 0
        self.niter = 0
    def __call__(self, relres):
        self.niter += 1
        self.relres = relres

def saveCSRData(matStruct, freqs, numProcs, csrFile, vecSubCutoffs=[], vecBasisSubDs=[], scalSubCutoffs=[], scalBasisSubDs=[]):
  """
  Saves matrix data in a format to be used by PETSc in parallel

  Inputs:
    matStruct         - The matrix for the problem
    freqs             - Frequencies to solve the problem at
    numProcs          - The number of processors to be used by PETSc
    csrFile           - The filename used to save the data for PETSc
    vecSubCutoffs     - Cutoffs in vector unknown numbering between subdomains. If provided, this information is used
                        to determine the splitting of rows between processors. If not, rows are split for even load balancing
    vecBasisSubDs     - Lists of vector unknowns associated with each subdomain, used for domain decomposition-based preconditioners in PETSc
    scalSubCutoffs    - Cutoffs in scalar unknown numbering between subdomains. If provided, this information is used
                        to determine the splitting of rows between processors. If not, rows are split for even load balancing
    scalBasisSubDs    - Lists of scalar unknowns associated with each subdomain, used for domain decomposition-based preconditioners in PETSc
  """

  if "Gnn" in matStruct:
    simType = "A"
  else:
    simType = "E"

  vecRowsPerProc, vecProcStartInds, procsPerSub, procSubDs = getProcSplit(numProcs, matStruct["K1"].shape[0], vecSubCutoffs) # Splitting for vector unknowns
  if simType == "A": # If scalar unknowns are present (for A-phi), get that splitting as well
    scalRowsPerProc, scalProcStartInds, _, _ = getProcSplit(numProcs, matStruct["Gnn"].shape[0], scalSubCutoffs)
  else:
    scalRowsPerProc = []

  matArr = [{} for _ in range(numProcs)]
  for key in matStruct:
    if type(matStruct[key]) == spa._csr.csr_array:
      if key[1] == "n": # Use scalar numbering for matrices with rows corresponding to scalar unknowns
        matArr = matSciPyToPETSc(matStruct[key], matArr, key, numProcs, scalProcStartInds)
      else:
        matArr = matSciPyToPETSc(matStruct[key], matArr, key, numProcs, vecProcStartInds) 
    elif type(matStruct[key]) == np.ndarray:
      matArr = vecSciPyToPETSc(matStruct[key][:matStruct["K1"].shape[0]], matArr, key+"_v", numProcs, vecProcStartInds) # Only has the vector part
      if simType == "A": # If in A-phi, also form the scalar part
        matArr = vecSciPyToPETSc(matStruct[key][matStruct["K1"].shape[0]:], matArr, key+"_s", numProcs, scalProcStartInds) # Only has the scalar part

  if simType == "A":
    # Form a mapping between unknowns of each type and the overall numbering
    vecTotMap = np.zeros((np.max(vecProcStartInds),), dtype=np.int32)
    scalTotMap = np.zeros((np.max(scalProcStartInds),), dtype=np.int32)
    for proc in range(numProcs):
      scalTotMap[scalProcStartInds[proc]:scalProcStartInds[proc+1]] = np.arange(scalProcStartInds[proc], scalProcStartInds[proc+1]) + vecProcStartInds[proc]
      vecTotMap[vecProcStartInds[proc]:vecProcStartInds[proc+1]] = np.arange(vecProcStartInds[proc], vecProcStartInds[proc+1]) + scalProcStartInds[proc+1]
  else:
    vecTotMap = np.arange(np.max(vecProcStartInds), dtype=np.int32)

  # Split the vector and scalar unknowns for each subdomain up between processors
  numSubDs = len(vecBasisSubDs)
  if numSubDs > 0:
    curSub, curSubProc = [0, 0] # The current subdomain and processor number within that subdomain
    for procNum in range(numProcs):
      if procsPerSub[curSub] == curSubProc:
        curSub += 1
        curSubProc = 1
      else:
        curSubProc += 1
      
      curVecStartInd, curVecEndInd = [vecProcStartInds[procNum], vecProcStartInds[procNum+1]]
      if simType == "A":
        curScalStartInd, curScalEndInd = [scalProcStartInds[procNum], scalProcStartInds[procNum+1]]

      # # Construct overlap lists relating subdomains
      matArr[procNum]["vecBasisSubDs"] = [[] for _ in range(numSubDs)]
      for subNum in range(numSubDs):
        # Loop through each subdomain and check if this processor contains any overlapping entries
        curSubVecOv = vecBasisSubDs[curSub][subNum]
        matArr[procNum]["vecBasisSubDs"][subNum] = curSubVecOv[np.all([curSubVecOv >= curVecStartInd, curSubVecOv < curVecEndInd], axis=0)]

      if simType == "A":
        matArr[procNum]["scalBasisSubDs"] = [[] for _ in range(numSubDs)]
        for subNum in range(numSubDs):
          # Loop through each subdomain and check if this processor contains any overlapping entries
          curSubScalOv = scalBasisSubDs[curSub][subNum]
          matArr[procNum]["scalBasisSubDs"][subNum] = curSubScalOv[np.all([curSubScalOv >= curScalStartInd, curSubScalOv < curScalEndInd], axis=0)]

  # Build a combined list of basis subdomains
  if simType == "A":
    for proc in range(numProcs):
      matArr[proc]["totBasisSubDs"] = [[] for _ in range(numSubDs)]
      for subNum in range(numSubDs):
        matArr[proc]["totBasisSubDs"][subNum] = np.append(scalTotMap[matArr[proc]["scalBasisSubDs"][subNum]], \
                                                               vecTotMap[matArr[proc]["vecBasisSubDs"][subNum]])

  resDict = {"vecLocSizes": np.array(vecRowsPerProc), "scalLocSizes": np.array(scalRowsPerProc), "procSubDs": procSubDs}
  if len(freqs) != 0: # FD Simulation
    resDict["freqs"] = freqs

  if numSubDs != 0:
    # Create a list of all unknowns belonging to each subdomain
    if simType == "E":
      for subNum in range(numSubDs):
        resDict[f"totSubD_{subNum}"] = vecBasisSubDs[0][subNum]
        for ii in range(1, numProcs):
          resDict[f"totSubD_{subNum}"] = np.append(resDict[f"totSubD_{subNum}"], vecBasisSubDs[ii][subNum])
    elif simType == "A":
      for subNum in range(numSubDs):
        resDict[f"totSubD_{subNum}"] = matArr[0]["totBasisSubDs"][subNum]
        for ii in range(1, numProcs):
          resDict[f"totSubD_{subNum}"] = np.append(resDict[f"totSubD_{subNum}"], matArr[ii]["totBasisSubDs"][subNum])
      
  np.savez(csrFile, *matArr, **resDict)
  
def matSciPyToPETSc(A, matArr, key, numProcs, procStartInds):
  """
  Converts the format of a SciPy matrix to a set of dictionaries 
  that can be imported into PETSc 

  Inputs:
    A             - Matrix to convert
    matArr        - List containing the prior converted matrices
    key           - String describing the current matrix
    numProcs      - Number of processors the matrix is split between
    procStartInds - Range of row indices of the matrix owned by each processor

  Outputs:
    matArr        - An updated version of matArr including this matrix
  """

  ## Extract the CSR data from the matrix
  rowptr = A.indptr.astype(np.int32)
  colval = A.indices.astype(np.int32)
  data = A.data      
  
  ## Then form an array to split up the data and fill it
  for proc in range(numProcs):
    matArr[proc][key] = {}
    matArr[proc][key]["rowptr"] = rowptr[procStartInds[proc]:procStartInds[proc+1]+1]
    matArr[proc][key]["colval"] = colval[matArr[proc][key]["rowptr"][0]:matArr[proc][key]["rowptr"][-1]]
    matArr[proc][key]["data"] = data[matArr[proc][key]["rowptr"][0]:matArr[proc][key]["rowptr"][-1]]

  return matArr

def vecSciPyToPETSc(b, matArr, key, numProcs, procStartInds):
  """
  Converts the format of a NumPy vector to a set of dictionaries 
  that can be imported into PETSc 

  Inputs:
    b             - Vector to convert
    matArr        - List containing the prior converted matrices and vectors
    key           - String describing the current vector
    numProcs      - Number of processors the vector is split between
    procStartInds - Range of row indices of the vector owned by each processor

  Outputs:
    matArr        - An updated version of matArr including this vector
  """
  
  # b_dict = {}
  for proc in range(numProcs):
    # b_dict[f"proc{proc}"] = b[procStartInds[proc]:procStartInds[proc+1]]
    matArr[proc][key] = b[procStartInds[proc]:procStartInds[proc+1]]

  return matArr

def getProcSplit(numProcs, numRows, subCutoffs=[]):
  """
  Determines the splitting of rows among processors for parallel operations in PETSc

  Inputs:
    numProcs   - Total number of processors available
    numRows    - Total number of rows in the matrix
    subCutoffs - Subdomain cutoffs (for when domain decomposition is used)

  Outputs:
    rowsPerProc   - Number of rows owned by each processor
    procStartInds - Row indices each processors ownership starts from
    procsPerSub   - Number of processors associated with each subdomain (when domain decomposition is used)
    procSubDs     - Subdomain each processor belongs to
  """

  ## Determine how many rows belong to each process
  numSubs = len(subCutoffs)
  if numSubs > 0: # If subdomains are used, based the splitting on them
    if numSubs == numProcs: # 1 subdomain per process
      procStartInds = np.append(0, subCutoffs)
      rowsPerProc = [procStartInds[ii+1] - procStartInds[ii] for ii in range(len(procStartInds)-1)]
      procsPerSub = np.ones((numSubs,), dtype=np.int32)
      procSubDs = np.arange(numSubs) # Marks the subdomain number for each process
    elif numSubs < numProcs: # Multiple processes per subdomain
      # First determine how many processors belong to each subdomain
      procsPerSub = np.ones((numSubs,), dtype=np.int32)*int(np.floor(numProcs/numSubs))
      leftoverProcs = np.mod(numProcs, numSubs)
      procsPerSub[:leftoverProcs] += 1

      # Now split unknowns between processors within each subdomain
      procStartInds = [0]
      procSubDs = [] # Marks the subdomain number for each process
      for ii in range(numSubs):
        subStart = procStartInds[-1] # First unknown for the subdomain
        for jj in range(procsPerSub[ii]):
          procStartInds.append(subStart + np.round((jj+1)*(subCutoffs[ii]-subStart)/procsPerSub[ii]))
          procSubDs.append(ii)
      procStartInds = np.array(procStartInds, dtype=np.int32)
      procSubDs = np.array(procSubDs, dtype=np.int32)
      rowsPerProc = [procStartInds[ii+1] - procStartInds[ii] for ii in range(len(procStartInds)-1)]

      # assert False, "Multiple processes per subdomain are not currently supported"
    elif len(subCutoffs) > numProcs: # Multiple subdomains per process
      assert False, "Multiple subdomains per process are not currently supported"

  else: # Otherwise, balance the loads equally
    procStartInds = np.arange(np.floor(numRows/numProcs), numRows+1, np.floor(numRows/numProcs), dtype=np.int32) # Entries assigned to each processor
    spareRows = np.mod(numRows, numProcs) # Assign extra rows to the first few processes if things don't divide evenly
    offset = 0
    while spareRows > 0:
      procStartInds[offset:] += 1
      offset += 1
      spareRows -= 1
    procStartInds = np.insert(procStartInds, 0, 0)
    rowsPerProc = [procStartInds[ii+1] - procStartInds[ii] for ii in range(len(procStartInds)-1)]
    procsPerSub = []
    procSubDs = []

  return rowsPerProc, procStartInds, procsPerSub, procSubDs

def modify_f_vector(matStruct, port_idx, basisMaps):  
  """
  Function to alter the driving term for simulations driving from multiple ports

  Inputs:
    matStruct - Dictionary containing each of the matrices
    port_idx  - Index of the current port to drive from
    basisMaps - Contains mappings between basis functions and geometric entities
  Outputs:
    matStruct - matStruct with the driving vector updated

  """
  voltVecInds = basisMaps["portVoltVecInds"]
  voltVecVals = basisMaps["portVoltVecVals"]
  
  matStruct["f"][:] = 0 # Set all values of f to zero
  matStruct["f"][voltVecInds[port_idx]] = -voltVecVals[port_idx]
  
  return matStruct

def matrix_remap(A, axis):
  """
  A function to remap a sparse matrix into a condensed form removing all zero rows / columns

  Inputs:
    A      - The original sparse matrix
    axis   - The axis to condense (0 for rows, 1 for columns)

  Outputs:
    A_cond - The condensed form of the sparse matrix
    P      - A sparse matrix used to permute the condensed form back into the original.
             If axis == 0, A = P @ A_cond. Else, A = A_cond @ P.
  """

  A_coo = A.asformat("coo")
  if axis == 0:
    rowInds = np.unique(A_coo.row)
    rowMap = np.arange(len(rowInds))
    newRows = np.zeros(len(A_coo.row), dtype=np.int32)
    for ii in range(len(newRows)):
      newRows[ii] = rowMap[A_coo.row[ii] == rowInds]
    
    A_cond = spa.csr_array((A_coo.data, (newRows, A_coo.col)), shape=(len(rowInds), A_coo.shape[1]), dtype=A_coo.dtype)
    P = spa.csr_array((np.ones(len(rowInds), dtype=np.int32), (rowInds, rowMap)), (A_coo.shape[0], len(rowInds)), dtype=np.int32)
  elif axis == 1:
    colInds = np.unique(A_coo.col)
    colMap = np.arange(len(colInds))
    newCols = np.zeros(len(A_coo.col), dtype=np.int32)
    for ii in range(len(newCols)):
      newCols[ii] = colMap[A_coo.col[ii] == colInds]

    A_cond = spa.csr_array((A_coo.data, (A_coo.row, newCols)), shape=(A_coo.shape[0], len(colInds)), dtype=A_coo.dtype)
    P = spa.csr_array((np.ones(len(colInds), dtype=np.int32), (colMap, colInds)), (len(colInds), A_coo.shape[1]), dtype=np.int32)

  return A_cond, P

def match_rank(A, tol):
  """
  A function for removing linearly dependent columns of a sparse CSR matrix
  Inputs:
    A       - The sparse CSR matrix to be modified
    tol     - Tolerance to determine whether a matrix is linearly dependent

  Outputs:
    A_red   - The reduced form of A, with linearly dependent columns removed
    scalMat - Matrix indicating which columns were dependent on which, and the ratio between those columns
    matRank - Rank of the matrix (equal to the number of columns in reduced form)
  """

  numCols = A.shape[1]
  scalMat = np.zeros((numCols, numCols))
  matRank = 0 # Track the matrix rank
  for ii in range(A.shape[1]):
    if np.any(scalMat[:,ii] != 0):
      continue # Don't need to look at this column if it's dependent on a prior column

    scalMat[ii,ii] = 1 # Indicate that this column is considered independent
    matRank += 1
    nnz = A[:,[ii]].nnz
    curData = A[:,[ii]].data
    curNorm = np.linalg.norm(curData)
    for jj in range(ii+1,numCols):
      if A[:,[jj]].nnz != nnz or np.any(A[:,[ii]].indptr != A[:,[jj]].indptr): 
        continue

      # Check if column jj is a scaled version of column ii
      scalVal = A[:,[jj]].data[0]/curData[0]
      if np.linalg.norm(curData - scalVal*A[:,[jj]].data)/curNorm < tol: 
        scalMat[ii,jj] = scalVal # If yes, record the scaling value

  # Construct the new matrix
  keepCols = np.where(scalMat.diagonal != 0)[0]
  A_red = A[:,keepCols]

  return A_red, scalMat, matRank


def spai_full(A, lfil, maxit=-1, realVal=False):
  '''
  A function for calculating the sparse approximate inverse (SPAI) of a sparse matrix A. 
  
  Inputs:
    A       - The sparse matrix to calculate the SPAI for
    lfil    - Number of non-zero elements filled in each row
    maxit   - Maximum number of iterations for the inner loop
    realVal - indicates whether the values of A are real (True) or complex (False)

  Outputs:
    M - The SPAI of A (as a sparse matrix)
  '''

  if realVal:
    dtype = np.float_
  else:
    dtype = np.complex128

  ## Start by scaling A such that it has unit diagnoals and norm(A) = 1
  D = spa.diags_array(1/np.sqrt(A.diagonal()), offsets=0) # Isolate and invert the diagonal of A
  A_temp = D @ spa.tril(A, -1) @ D
  n = A.shape[0]
  I = spa.eye_array(n, n) # Sparse identity matrix with the same size as A
  A_scal = A_temp + A_temp.conj().transpose() + I # scaled version of a
  A_scal_cols = [A_scal[:,[ii]] for ii in range(n)]
  #A_scal = A_scal.tocsc() # Swap to csc to allow for easy column slicing

  ## Now calculate the SPAI of normalized A
  if maxit == -1: # Assign default if no input is provided
    maxit = 4*lfil

  numEntries = lfil*n # Number of nonzero entries in M
  rows = np.zeros((numEntries,))
  cols = np.zeros((numEntries,))
  M_vals = np.zeros((numEntries,), dtype=dtype)
  m0 = np.zeros((n,1), dtype=dtype)

  for ii in range(n):
    #m = spa.csc_array(([1], ([ii],[0])), (n,1), dtype=dtype)
    m = m0.copy()
    m[ii] = 1
    r = m - A_scal_cols[ii] # Residual
    nnz = 1
    #start_time = time.perf_counter_ns()
    for jj in range(1,maxit):
      ind = np.argmax(np.abs(r))
      if m[ind][0] == 0:
        nnz += 1
      delta = r[ind][0]
      m[ind] += delta
      if nnz >= lfil:
        break
      r -= delta*A_scal_cols[ind]
    #print(f"Inner loop iteration time: {(time.perf_counter_ns() - start_time)/1e9} s")

    nzInds = np.where(m != 0)[0]
    if nnz != lfil: # Handle the case where fewer than lfil entries were used
      rows_new = np.ones((lfil,))*-1 # Mark unused indices with -1 so they can be removed
      rows_new[0:nnz] = nzInds
      cols_new = np.zeros((lfil,))
      cols_new[0:nnz] = ii
      M_vals_new = np.zeros((lfil,), dtype=dtype)
      M_vals_new[0:nnz] = m[nzInds][:,0]
    else:
      rows_new = nzInds
      cols_new = ii*np.ones((lfil,))
      M_vals_new = m[nzInds][:,0]

    rows[(ii*lfil):((ii+1)*lfil)] = rows_new
    cols[(ii*lfil):((ii+1)*lfil)] = cols_new
    M_vals[(ii*lfil):((ii+1)*lfil)] = M_vals_new

  nonzeroInds = np.where(rows != -1)
  rows = rows[nonzeroInds]
  cols = cols[nonzeroInds]
  M_vals = M_vals[nonzeroInds]
  M = spa.csr_array((M_vals, (rows, cols)), (n,n), dtype=dtype)

  ## Make M exactly symmetric
  d = M.diagonal()
  M = (M+M.conj().transpose())/2 # Enforce exact hermicity
  M = spa.tril(M, -1)
  M = M + M.conj().transpose() + spa.diags(d).tocsr()

  ## Reverse normalization
  M_u = spa.triu(M, 1).tocsr()
  M_l = spa.tril(M, -1).tocsr()
  M_d = spa.diags(M.diagonal()).tocsr()

  M_u_new = spa.dok_array((n,n), dtype=dtype)
  M_l_new = spa.dok_array((n,n), dtype=dtype)
  M_d_new = spa.dok_array((n,n), dtype=dtype)

  for ii in range(n):
    nz_u = M_u[[ii],:].indices
    nz_l = M_l[[ii],:].indices
    
    for jj in nz_u:
      M_u_new[ii,jj] = M_u[[ii],[jj]]/np.sqrt(A[[ii],[ii]]*A[[jj],[jj]])
    
    for jj in nz_l:
      M_l_new[ii,jj] = M_l[[ii],[jj]]/np.sqrt(A[[ii],[ii]]*A[[jj],[jj]])
    
    M_d_new[ii,ii] = M_d[[ii],[ii]] / A[[ii],[ii]]
  
  M = M_u_new.tocsr() + M_d_new.tocsr() + M_l_new.tocsc()

  return M
