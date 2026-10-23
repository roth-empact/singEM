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
      matStruct["K1"] = K1 - matStruct["Ken"] @ Gnn_inv @ matStruct["Kne"]
      # matStruct["K1"] = K1 - matStruct["Ken"]._matmul_sparse(Gnn_inv._matmul_sparse(matStruct["Kne"])) # Replace K1 with the low frequency matrix
      print(f"Finished forming low frequency matrix. Current time: {tz.localize(datetime.now())}")
      stats[f"lfil={lfil}"]["fullNnz"] = matStruct["K1"].nnz
      stats[f"lfil={lfil}"]["fullMem"] = memFactor*stats[f"lfil={lfil}"]["fullNnz"]
      stats[f"lfil={lfil}"]["memIncrease"] = stats[f"lfil={lfil}"]["fullMem"]/stats[f"lfil={lfil}"]["blockMem"]
      np.savez("SPAI_stats.npz", stats=stats)

      print(f"Finished building SPAI for lfil = {lfil} in {(time.time() - start_time):.3f} s. Starting to solve the systems.")
    
      start_time = time.time()
      A, b, M_diag, _, _ = buildSysFD(matStruct, freqs, simType="E")
      # A_static = M_diag[0] @ matStruct["K1"] @ M_diag[0]
      # fillFactor = np.max([1, (solveP["rowFill"]/stats[f"lfil={lfils[ii]}"]["memIncrease"])])
      # A_static_ilu = spa.linalg.spilu(A_static.astype(np.complex128).tocsc(), drop_tol=solveP["dropTol"], fill_factor=fillFactor)
      # M = spa.linalg.LinearOperator(A_static.shape, A_static_ilu.solve)
      diagnostics = {}
      diagnostics["iter"] = np.zeros((len(freqs),), dtype=np.int32)
      x = np.zeros((N_e+N_n,len(freqs)), dtype=np.complex128)
      for jj in range(len(freqs)):
        b[jj] /= 1j*2*np.pi*freqs[jj]
        x[:N_e,jj] = itSolve_PETSc(A[jj], b[jj], solveP, numProcs)
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
    Gnn_inv_approx = spa.diags_array(1/matStruct["Gnn"].diagonal(), format='csr')
  stats["spaiNnz"] = Gnn_inv_approx.nnz
  stats["spaiTime"] = time.time() - spai_start

  # Assess the accuracy of the SPAI
  stats["invAccuracy"] = spa.linalg.norm(Gnn_inv_approx @ matStruct["Gnn"] - spa.eye_array(N_n), ord='fro')/spa.linalg.norm(spa.eye_array(N_n), ord='fro')

  return Gnn_inv_approx, stats

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
                      default='./Meshes/CPW/CPW_resonator_FD.json')
  parser.add_argument('-v', '--verbose', 
                      help="Prints details about simulation status to the command line.",
                      action="store_true")
  args = parser.parse_args()
    
  main(args.simFile, True)