import numpy as np
import scipy.sparse as spa
import sys
import time
sys.path.append('/PV_visualization/')
from PV_visualization.pvController import *
from mesh import initMesh
from meshToMats import meshToMats
from solveFD import solveFD, buildSysFD
import argparse
from JSON_util import readSimJSON
from linalg_funcs import spai_full, itSolve_par, itSolve_PETSc
from multiprocessing import Pool
from itertools import repeat
import postProcess as post
from datetime import datetime
from pytz import timezone
import os

# Inputs:
# sp - dictionary of simulation parameters
# portParams - dictionary of port parameters
def main(simFile, verbose):
  """
  Overall wrapper for finite element simulations. Supply with a JSON file
  for configuring the simulation, which can be generated with writeSimJSON().
  That file must reference a mesh file generated with Cubit, as well as a 
  geometry configuration file generated with writeGeometryJSON().

  Inputs:
    simFile - Path to JSON file containing simulation parameters and references 
              to the geometry and mesh files
    verbose - Indicates whether to provide additional printout information

  """
  tz = timezone('EST')

  ## Read in the initial configuration information
  simType, probP, solveP, outP, numProcs = readSimJSON(simFile) 
  if verbose:
    match simType:
      case "E":
        methodStr = "field-based method"
      case "A":
        methodStr = "potential-based method"
      case "TCS":
        methodStr = "tree-cotree splitting"
    match probP["type"]:
      case "FD":
        probStr = "frequency domain"
      case "TD":
        probStr = "time domain"
      case "Eig":
        probStr = "eigenmode"
    geomFile = probP["geomFile"]
    print(f"Using simulation settings from {simFile} and geometry settings from {geomFile}")
    meshFile = probP["meshFile"]
    print(f"Solving {probStr} problem on {meshFile} using {methodStr}.")

  # Prepare filenames for AMR (if used)
  if probP["maxAMR"] != 1:
    cubFile_base = probP["cubFile"][:probP["cubFile"].find(".",1)]
    meshFile_base = probP["meshFile"][:probP["meshFile"].find(".",1)]
    resFile = outP["resFile"]
    if resFile != None:
      resFileEnd = resFile.find(".",1)
      curResFile = f"{resFile[:resFileEnd]}_0{resFile[resFileEnd:]}"
    else:
      curResFile = None
  else:
    curResFile = outP["resFile"]
        
  for passNum in range(probP["maxAMR"]):
    pass_start = time.time()
    ## Produce connectivity lists from the mesh and geometry files
    connLists, meshStats = initMesh(probP["meshFile"], probP["geomFile"], simType, probP["outBC"])
    N_n, N_e, N_w_e, N_f, N_q_f = meshStats["N_n"], meshStats["N_e"], meshStats["N_w_e"], meshStats["N_f"], meshStats["N_q_f"]
    if verbose:
      print(f"Time spent forming element lists: {(time.time() - pass_start):.3f} s")
      numTets = len(connLists["tetList"])
      print(f"Number of tetrahdrons: {numTets}")
      numWedges = len(connLists["wedgeList"])
      print(f"Number of wedges: {numWedges}")
      print(f"Number of nodes: {N_n}")
      print(f"Number of edges: {N_e+N_w_e}")
      print(f"Number of triangular faces: {N_f}")
      print(f"Number of quadrilateral faces: {N_q_f}")

      l_max, l_min, l_avg, l_stdev = meshStats["l_max"], meshStats["l_min"], meshStats["l_avg"], meshStats["l_stdev"]
      print(f"Longest triangular edge: {(l_max*1e3):.3e} mm")
      print(f"Shortest triangular edge: {(l_min*1e3):.3e} mm")
      print(f"Average triangular edge length: {(l_avg*1e3):.3e} mm")
      print(f"Triangular edge length standard deviation: {(l_stdev*1e3):.3e} mm")

    ## Form matrices and driving vectors
    matStruct, basisMaps = meshToMats(connLists, probP["bOrder"], probP["3DIntOrder"], probP["2DIntOrder"], simType, probP["outBC"], meshStats, numProcs)
     
    ## Build G_NN^{-1} and estimate accuracy
    start_time = time.time()
    Gnn = matStruct["Gnn"]
    print(f"Gnn.nnz: {Gnn.nnz}")
    rowFill = np.round(Gnn.nnz/N_n)
    lfils = [1]#[1, 5, 10, 50, 100, 500, 1000]
    freqs = probP["freqs"]
    stats = {}
    outVals = {}
    stats["lfils"] = lfils
    K1 = matStruct["K1"].copy()
    matStruct["f"] = matStruct["f"][:N_e] # Reduce dimension of the source vector
    memFactor = (16+4)/(1024**3) # Factor for converting sparse matrix nonzeros to memory in GB
    for ii, lfil in enumerate(lfils):
      Gnn_inv, curStats = getSPAIStats(matStruct, lfil, N_n)
      stats[f"lfil={lfil}"] = {}
      for stat in curStats.keys():
        stats[f"lfil={lfil}"][stat] = curStats[stat]
      print(f"Starting to form low frequency matrix. Current time: {tz.localize(datetime.now())}")
      # matStruct["K1"] = K1 - matStruct @ Gnn_inv @ matStruct["Kne"]
      # matStruct["K1"] = K1 - matStruct["Ken"]._matmul_sparse(Gnn_inv._matmul_sparse(matStruct["Kne"])) # Replace K1 with the low frequency matrix
      # print(f"Finished forming low frequency matrix. Current time: {tz.localize(datetime.now())}")
      # stats[f"lfil={lfil}"]["fullNnz"] = matStruct["K1"].nnz
      # stats[f"lfil={lfil}"]["fullMem"] = memFactor*stats[f"lfil={lfil}"]["fullNnz"]
      # stats[f"lfil={lfil}"]["memIncrease"] = stats[f"lfil={lfil}"]["fullMem"]/stats[f"lfil={lfil}"]["blockMem"]
      # np.savez("SPAI_stats.npz", stats=stats)

      print(f"Finished building SPAI for lfil = {lfil} in {(time.time() - start_time):.3f} s. Starting to solve the systems.")
    
      start_time = time.time()
      _, b, M_diag, _, _ = buildSysFD(matStruct, freqs, simType="E")
      # A_static = M_diag[0] @ matStruct["K1"] @ M_diag[0]
      # fillFactor = np.max([1, (solveP["rowFill"]/stats[f"lfil={lfils[ii]}"]["memIncrease"])])
      # A_static_ilu = spa.linalg.spilu(A_static.astype(np.complex128).tocsc(), drop_tol=solveP["dropTol"], fill_factor=fillFactor)
      # M = spa.linalg.LinearOperator(A_static.shape, A_static_ilu.solve)
      diagnostics = {}
      diagnostics["iter"] = np.zeros((len(freqs),), dtype=np.int32)
      x = np.zeros((N_e+N_n,len(freqs)), dtype=np.complex128)
      for jj in range(len(freqs)):
        b[jj] /= 1j*2*np.pi*freqs[jj]
        # x[:N_e,jj], diagnostics["iter"][jj] = itSolve_par(A[jj], b[jj], freqs[jj], solveP["tol"], solveP["restart"], solveP["maxit"], M=M, debug=True)
        # x[:N_e,jj] = itSolve_PETSc(A[jj], b[jj], solveP, numProcs)
        if jj == 0:
          x[:N_e,jj], nnz = itSolve_PETSc_subMats(matStruct["K1"], matStruct["K2"], matStruct["Ken"], matStruct["Kne"], Gnn_inv, matStruct["f"][:N_e], freqs[jj], numProcs)
          stats[f"lfil={lfil}"]["fullNnz"] = nnz
          stats[f"lfil={lfil}"]["fullMem"] = memFactor*nnz
          stats[f"lfil={lfil}"]["memIncrease"] = stats[f"lfil={lfil}"]["fullMem"]/stats[f"lfil={lfil}"]["blockMem"]
        else:
          x[:N_e,jj], _ = itSolve_PETSc_subMats(matStruct["K1"], matStruct["K2"], matStruct["Ken"], matStruct["Kne"], Gnn_inv, matStruct["f"][:N_e], freqs[jj], numProcs)
        x[:N_e,jj] = M_diag[jj] @ x[:N_e,jj]
        x[N_e:, jj] = (matStruct["Gnn_inv_approx"] @ (matStruct["Kne"] @ x[:N_e,jj]))/(1j*2*np.pi*freqs[jj])
      solve_time = time.time() - start_time
      print(f"Finished solving the system for lfil={lfils[ii]} in {solve_time:.3f} s")
      stats[f"lfil={lfil}"]["solveTime"] = solve_time

      ## Pull back the full solution and calculate port voltages  
      outVals[f"lfil={lfil}"] = {}
      outVals[f"lfil={lfil}"]["portV"] = post.getPortVoltages(x, basisMaps["portVoltVecInds"], basisMaps["portVoltVecVals"], connLists["portP"], matStruct["K1"].shape[0], simType, "FD", freqs=freqs)
      outVals[f"lfil={lfil}"]["portS"] = post.getSParams(connLists["portP"], outVals[f"lfil={lfil}"]["portV"])
      outVals[f"lfil={lfil}"]["freqs"] = freqs
      outVals[f"lfil={lfil}"]["diagnostics"] = diagnostics
    
      np.savez("SPAI_outputs.npz", stats=stats, outVals=outVals)


def getSPAIStats(matStruct, lfil, N_n):

  stats = {}

  ## Get memory usage for the block system
  memFactor = (16+4)/(1024**3) # Factor for converting sparse matrix nonzeros to memory in GB
  stats["blockNnz"] = matStruct["K2"].nnz + matStruct["Ken"].nnz + matStruct["Kne"].nnz + matStruct["Gnn"].nnz
  stats["blockMem"] = memFactor*stats["blockNnz"]

  # Form the SPAI and get memory usage for it
  spai_start = time.time()
  if lfil != 1:
    Gnn_inv_approx = spai_full(matStruct["Gnn"], lfil, realVal=True)
  else: # Use diagonal approximation
    Gnn_inv_approx = spa.diags_array(1/np.abs(matStruct["Gnn"].diagonal()), format='csr')
  stats["spaiNnz"] = Gnn_inv_approx.nnz
  stats["spaiTime"] = time.time() - spai_start

  # Assess the accuracy of the SPAI
  stats["invAccuracy"] = spa.linalg.norm(Gnn_inv_approx @ matStruct["Gnn"] - spa.eye_array(N_n), ord='fro')/spa.linalg.norm(spa.eye_array(N_n), ord='fro')

  return Gnn_inv_approx, stats

def itSolve_PETSc_subMats(K1, K2, Ken, Kne, Gnn_inv, b, freq, numProcesses):
  """
  Sets up data to build the full matrix in PETSc and solve there
  """
  ## Extract the CSR data from each matrix
  for ii, mat in enumerate([K1, K2, Ken, Kne, Gnn_inv]):
    rowptr = mat.indptr.astype(np.int32)
    colval = mat.indices.astype(np.int32)
    data = mat.data      

    rowptr_dict = {}
    colval_dict = {}
    data_dict = {}
    nv_proc = np.zeros((numProcesses,), dtype=np.int32) # Number of nonzero values per processor

    ## First determine how many rows belong to each process
    rowsPerProc = np.arange(np.floor((len(rowptr)-1) / numProcesses), len(rowptr), np.floor((len(rowptr)-1) / numProcesses), dtype=np.int32) # Entries assigned to each processor
    spareRows = np.mod(len(rowptr)-1, numProcesses) # Assign extra rows to the first few processes if things don't divide evenly
    offset = 0
    while spareRows > 0:
      rowsPerProc[offset:] += 1
      offset += 1
      spareRows -= 1
  
    ## Then form an array to split up the data and fill it
    for proc in range(numProcesses):
      if proc == 0:
        rowptr_dict[f"proc{proc}"] = rowptr[:rowsPerProc[proc]+1]
      else:
        rowptr_dict[f"proc{proc}"] = rowptr[rowsPerProc[proc-1]:rowsPerProc[proc]+1]
      colval_dict[f"proc{proc}"] = colval[rowptr_dict[f"proc{proc}"][0]:rowptr_dict[f"proc{proc}"][-1]]
      data_dict[f"proc{proc}"] = data[rowptr_dict[f"proc{proc}"][0]:rowptr_dict[f"proc{proc}"][-1]]
      nv_proc[proc] = rowptr_dict[f"proc{proc}"][-1]-rowptr_dict[f"proc{proc}"][0]
    
    if ii == 0:
      K1_dict = {'rowptr': rowptr_dict, 'colval': colval_dict, 'data': data_dict, 'nv_proc': nv_proc}
    elif ii == 1:
      K2_dict = {'rowptr': rowptr_dict, 'colval': colval_dict, 'data': data_dict, 'nv_proc': nv_proc}
    elif ii == 2:
      Ken_dict = {'rowptr': rowptr_dict, 'colval': colval_dict, 'data': data_dict, 'nv_proc': nv_proc}
    elif ii == 3:
      Kne_dict = {'rowptr': rowptr_dict, 'colval': colval_dict, 'data': data_dict, 'nv_proc': nv_proc}
    elif ii == 4:
      Gnn_dict = {'rowptr': rowptr_dict, 'colval': colval_dict, 'data': data_dict, 'nv_proc': nv_proc}
   
  b_dict = {}
  ## First determine how many rows belong to each process
  rowsPerProc = np.arange(np.floor((len(b)-1) / numProcesses), len(b), np.floor((len(b)-1) / numProcesses), dtype=np.int32) # Entries assigned to each processor
  spareRows = np.mod(len(b)-1, numProcesses) # Assign extra rows to the first few processes if things don't divide evenly
  offset = 0
  while spareRows > 0:
    rowsPerProc[offset:] += 1
    offset += 1
    spareRows -= 1

  ## Then form an array to split up the data and fill it
  for proc in range(numProcesses):
    if proc == 0:
      b_dict[f"proc{proc}"] = b[:rowsPerProc[proc]]
    else:
      b_dict[f"proc{proc}"] = b[rowsPerProc[proc-1]:rowsPerProc[proc]]

  ## Write the vector and csr data to a file
  csrFile = "csrData.npz"
  np.savez(csrFile, K1=K1_dict, K2=K2_dict, Ken=Ken_dict, Kne=Kne_dict, Gnn_inv=Gnn_dict, b=b_dict, N_e=Ken.shape[0], N_n=Ken.shape[1])

  ## Solve the system in parallel using SLEPc with MPI
  cmd = ["mpiexec", "-n", f"{numProcesses}", "python", "PETSc_solve_subMats.py", "-n", f"{len(rowptr)-1}", "-f", csrFile, "--freq", f"{freq}", "-ksp_type", "preonly", "-pc_type", "lu"]
  result = subprocess.run(cmd, capture_output=True, text=True)
  print(result.stdout)
  print(result.stderr)

  ## Load in the results and eliminate the temp files
  resFile = "PETSc_result.npz"
  res = np.load(resFile, allow_pickle=True)
  totVals = res['x']
  nnz = res['nnz']
  os.remove(csrFile)
  os.remove(resFile)

  return totVals, nnz

if __name__ == "__main__":
  #* Script Arguments *#
  parser = argparse.ArgumentParser(
                  prog='main.py',
                  description='Main wrapper for finite element simulations.')
  # parser.add_argument('-f', '--simFile',
  #                     help=".json simulation file. See writeSimJSON() for format details.",
  #                     default='./Meshes/Prisms/Cavities/prismCavity_doubleLayer_Eig.json')
  parser.add_argument('-f', '--simFile',
                      help=".json simulation file. See writeSimJSON() for format details.",
                      default='./Meshes/Microstrips/microstripFD.json')
  parser.add_argument('-v', '--verbose', 
                      help="Prints details about simulation status to the command line.",
                      action="store_true")
  args = parser.parse_args()
    
  main(args.simFile, True)