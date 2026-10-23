import numpy as np
import sys
import petsc4py #type:ignore
import slepc4py #type:ignore
petsc4py.init(sys.argv)
from petsc4py import PETSc #type:ignore
from slepc4py import SLEPc #type:ignore
import sys, time

if __name__ == "__main__":

  ## Create vector for storing matrix data in parallel
  OptDB = PETSc.Options()

  filename = OptDB.getString('f', "")
  # filename = "csrData.npz"
  try:
    matData = np.load(filename, mmap_mode='r', allow_pickle=True)
  except:
    print("A valid filename for CSR values to be used in PETSc was not provided.")
    raise Exception
  
  ## Set up parallel process
  comm = PETSc.COMM_WORLD
  numProcs = comm.getSize()
  if comm.getRank() == 0:
    print(f"Number of PETSc processes: {numProcs}.")
  
  n = OptDB.getInt('n', 0) # Number of rows in the matrix
  sigma = OptDB.getReal('sigma', 0) # Shift for shift-invert mode
  neig = OptDB.getInt('neig', 0) # Number of eigenpairs to find
  which = OptDB.getString('which', "SM")
  # n = 223668
      
  ## Build the left-hand side matrix in PETSc
  A = PETSc.Mat().create(comm=comm)
  A.setSizes([n, n])
  # A.setType("mpimaij")

  rowptr = matData["A"].item()["rowptr"][f"proc{comm.getRank()}"]
  rowptr = rowptr - rowptr[0]
  colval = matData["A"].item()["colval"][f"proc{comm.getRank()}"]
  data = matData["A"].item()["data"][f"proc{comm.getRank()}"]

  A.setValuesCSR(rowptr, colval, data)
  A.assemblyBegin()
  A.assemblyEnd()	

  ## Build the right-hand side matrix in PETSc
  B = PETSc.Mat().create(comm=comm)
  B.setSizes([n, n])

  rowptr_B = matData["B"].item()["rowptr"][f"proc{comm.getRank()}"]
  rowptr_B = rowptr_B - rowptr_B[0]
  colval_B = matData["B"].item()["colval"][f"proc{comm.getRank()}"]
  data_B = matData["B"].item()["data"][f"proc{comm.getRank()}"]

  B.setValuesCSR(rowptr_B, colval_B, data_B)
  B.assemblyBegin()
  B.assemblyEnd()	

  ## Set up the solver
  eps = SLEPc.EPS()
  eps.create(comm=A.getComm())
  eps.setType('krylovschur')
  eps.setOperators(A, B)
  eps.setDimensions(nev=neig)

  ## If a shift is provided, then apply it
  if sigma != 0:
    st = eps.getST()
    st.setShift(sigma)
    st.setType('sinvert')
    eps.setWhichEigenpairs(which=SLEPc.EPS.Which.TARGET_MAGNITUDE)
  else:
    if which == "LM":
      eps.setWhichEigenpairs(which=SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    elif which == "SM":
      eps.setWhichEigenpairs(which=SLEPc.EPS.Which.SMALLEST_MAGNITUDE)

  ## Find the eigenpairs
  start_time = time.time()
  eps.solve()
  nconv = eps.getConverged()
  if nconv > neig:
    nconv = neig
  if comm.getRank() == 0:
    print(f"Number of converged eigenpairs: {nconv}")
  
  ## Output the eigenpairs to a file
  if comm.getRank() == 0:
    eigvals = np.zeros((nconv,), dtype=np.complex128)
    eigvecs = np.zeros((n,nconv), dtype=np.complex128)
  eigInd = 0
  for ii in range(nconv):
    # Get the current eigenvector and value
    vr, _ = A.getVecs() # Real part of eigenvector
    vi, _ = A.getVecs() # Imaginary part of eigenvector
    k = eps.getEigenpair(ii, vr, vi)

    ## Check if the eigenvalue is invalid (large imaginary component or near zero)
    if np.abs(np.imag(k)) > np.abs(1e-3*np.real(k)):
      continue
    elif sigma != 0 and np.abs(k) < 1e-3*sigma:
      continue

    # Move the vectors to processor 0
    (sct, eigv_re) = PETSc.Scatter().toZero(vr)
    sct.scatter(vr, eigv_re)
    (sct, eigv_im) = PETSc.Scatter().toZero(vi)
    sct.scatter(vi, eigv_im)
    if comm.getRank() == 0:
      eigvals[eigInd] = k
      eigvecs[:,eigInd] = eigv_re.array + 1j*eigv_im.array
    eigInd += 1
    
  if comm.getRank() == 0:
    print(f"SLEPc solve time: {(time.time() - start_time):.5f} s.")
    print(f"{ii - eigInd} invalid eigenpairs were found.")
    np.savez("SLEPc_result.npz", eigvals=eigvals, eigvecs=eigvecs)
