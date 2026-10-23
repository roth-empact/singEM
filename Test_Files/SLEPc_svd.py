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
  nsv = OptDB.getInt('nsv', 0) # Number of eigenpairs to find
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

  ## Set up the solver
  svd = SLEPc.SVD()
  svd.create(comm=A.getComm())
  svd.setType('cyclic')
  svd.setOperators(A)
  svd.setDimensions(nsv=nsv)
  if which == "LM":
    svd.setWhichSingularTriplets(which=SLEPc.SVD.Which.LARGEST)
  elif which == "SM":
    svd.setWhichSingularTriplets(which=SLEPc.SVD.Which.SMALLEST)

  ## Find the singular values
  start_time = time.time()
  svd.solve()
  nconv = svd.getConverged()
  if nconv > nsv:
    nconv = nsv
  if comm.getRank() == 0:
    print(f"Number of converged singular values: {nconv}")
  
  ## Output the eigenpairs to a file
  if comm.getRank() == 0:
    svals = np.zeros((nconv,), dtype=np.float64)
  for ii in range(nconv):
    # Get the singular value
    sval = svd.getSingularTriplet(ii)
    if comm.getRank() == 0:
      print(f"Singular value {ii}: {sval:.5e}")
      svals[ii] = sval
    
  if comm.getRank() == 0:
    print(f"SLEPc solve time: {(time.time() - start_time):.5f} s.")
    np.savez("SLEPc_result.npz", svals=svals)
