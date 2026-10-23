import numpy as np
import sys
import petsc4py #type:ignore
petsc4py.init(sys.argv)
from petsc4py import PETSc #type:ignore
import sys, time

if __name__ == "__main__":

  ## Create vector for storing matrix data in parallel
  OptDB = PETSc.Options()

  # filename = OptDB.getString('f', "")
  filename = "csrData.npz"
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
  freq = OptDB.getReal('freq', 0)
  N_e = matData["N_e"].item()
  N_n = matData["N_n"].item()
  # n = 223668
      
  ## Build the matrix in PETSc
  matDict = {}
  matList = ["K1", "K2", "Ken", "Kne", "Gnn_inv"]
  matSizes = [[N_e, N_e], [N_e, N_e], [N_e, N_n], [N_n, N_e], [N_n, N_n]]
  for ii, matStr in enumerate(["K1", "K2", "Ken", "Kne", "Gnn_inv"]):
    matDict[matStr] = PETSc.Mat().create(comm=comm)
    matDict[matStr].setSizes([matSizes[ii][0], matSizes[ii][1]])

    rowptr = matData[matStr].item()["rowptr"][f"proc{comm.getRank()}"]
    rowptr = rowptr - rowptr[0]
    colval = matData[matStr].item()["colval"][f"proc{comm.getRank()}"]
    data = matData[matStr].item()["data"][f"proc{comm.getRank()}"]

    if comm.getRank() == 0:
      print(f"rowptr: {len(rowptr)}")
      print(f"colval: {len(colval)}")
      print(f"data: {len(data)}")
      print(matDict[matStr])
      print(f"Local size: {matDict[matStr].getLocalSize()}")
      print(f"Global size: {matDict[matStr].getSize()}")


    matDict[matStr].setValuesCSR(rowptr, colval, data)
    matDict[matStr].assemblyBegin()
    matDict[matStr].assemblyEnd()	

  # A = PETSc.Mat().create(comm=comm)
  # A.setSizes(N_e, N_e)
  A = matDict["K1"] - (2*np.pi*freq)**2*matDict["K2"] - matDict["Ken"].matMatMult(matDict["Gnn_inv"], matDict["Kne"])
  info = A.getInfo()
  np.savez("petscInfo.npz", info=info)

  # ## Set up the driving and solution vector
  b = A.createVecLeft()
  ownRange = b.owner_range
  b.setValues(np.arange(ownRange[0], ownRange[1], dtype=np.int32), matData["b"].item()[f"proc{comm.getRank()}"])
  b.assemblyBegin()
  b.assemblyEnd()
  x = A.createVecRight()

  ## Set up the solver
  ksp = PETSc.KSP().create(comm=A.getComm())
  ksp.setOperators(A)
  ksp.setConvergenceHistory()

  ksp.setFromOptions()
  pc = ksp.getPC()
  pc.setFromOptions()
  
  # Solve direct with LU
  # ksp.setType('preonly')
  # pc = ksp.getPC()
  # pc.setType('lu')
  
  # ## Solve with GMRES
  # ksp.setType('gmres')
  # ksp.setTolerances(rtol=1e-6, max_it=10000)
  # ksp.setGMRESRestart(30)
  
  # ## Precondition with point-block Jacobi
  # pc = ksp.getPC()
  # pc.setType('pbjacobi')
  
  # ## Precondition with the additive Schwarz method
  # pc = ksp.getPC()
  # pc.setType('asm')
  # pc.setASMType(PETSc.PC.ASMType.RESTRICT)
  # pc.setASMOverlap(10)
  # pc.setUp()
  # subKsp = pc.getASMSubKSP()[0]
  # subKsp.setType('preonly')
  # subPC = subKsp.getPC()
  # subPC.setType('lu')
  
  # ## Precondition with AMG
  # pc = ksp.getPC()
  # pc.setType('gamg')    
  
  ## Solve the problem
  start_time = time.time()
  ksp.solve(b,x)
  
  res = A*x - b
  itNum = ksp.getIterationNumber()
  # print(f"Process {comm.getRank()} relative residual: {(res.norm()/b.norm()):.5e}. Number of iterations: {itNum}.")
  
  ## Save the global solution
  (sct, globX) = PETSc.Scatter().toZero(x)
  sct.scatter(x, globX)
  if comm.getRank() == 0:
    globX = globX.array
    print(f"PETSc solve time: {(time.time() - start_time):.5f} s.")
    np.savez("PETSc_result.npz", x=globX, nnz=info['nz_used'])
