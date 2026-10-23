import numpy as np
import scipy.sparse as spa
import time
import subprocess, os

from postProcess import getPortVoltages, getTotField
from linalg_funcs import itSolve_par, saveCSRData
from solveFD import buildPETScCmd

def solveTD(matStruct, subDLists, connLists, basisMaps, probP, solveP, outP, simType, numProcs):
  """
  Outer wrapper for time-domain simulations
  """
  if solveP["pkg"].lower() == "scipy":
    portV, savedField, diagnostics = solveTD_SciPy(matStruct, connLists, basisMaps, probP, solveP, outP, simType)
  elif solveP["pkg"].lower() == "petsc":
    portV, savedField, diagnostics = solveTD_PETSc(matStruct, subDLists, connLists, basisMaps, probP, solveP, outP, simType, numProcs)

  return portV, savedField, diagnostics

def solveMS(matStruct, subDLists, connLists, basisMaps, probP, solveP, outP, simType, numProcs):
  if "vecSubCutoffs" in subDLists:
    vecSubCutoffs, vecSubDs = subDLists["vecSubCutoffs"], subDLists["vecSubDs"]
    scalSubCutoffs, scalSubDs = subDLists["scalSubCutoffs"], subDLists["scalSubDs"]
  else:
    vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs = [[], [], [], []]

  csrFile = "csrData_MS.npz"
  outFile = "PETSc_result_MS.npz"
  resFile = "PETSc_residual.npy"
  saveCSRData(matStruct, [], numProcs, csrFile, vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs, \
              Vs=probP["driveP"]["Vs"], portVoltVecInds=basisMaps["portVoltVecInds"], portVoltVecVals=basisMaps["portVoltVecVals"], \
              portVoltScalInds=basisMaps["portVoltScalInds"], portVoltScalVals=basisMaps["portVoltScalVals"],
              qubitVoltVecInds=basisMaps["qubitVoltVecInds"], qubitVoltVecVals=basisMaps["qubitVoltVecVals"], \
              qubitVoltScalInds=basisMaps["qubitVoltScalInds"], qubitVoltScalVals=basisMaps["qubitVoltScalVals"], qubitP=connLists["qubitP"])

  cmd = buildPETScCmd("PETSc_time_march_MS_qubitPorts.py", solveP, matStruct, csrFile, simType, vecSubCutoffs, numProcs)
  dt, numT, beta = [probP["dt"], probP["numT"], probP["beta"]]
  cmd.extend(["-dt", f"{dt}", "-numT", f"{numT}", "-beta", f"{beta}"])
  cmd.extend(["-snes_type", "qn", "-snes_qn_type", "lbfgs", "-snes_rtol", "1e-9", "-snes_divtol", "10"])
  if "coupling" in probP:
    cmd.extend(["-coupling", probP["coupling"].lower()])
  print(cmd)

  # Solve the problem in PETSc
  result = subprocess.run(cmd, capture_output=True, text=True)
  print(result.stdout)
  print(result.stderr)

  # Load solution and residual from files, delete files and return
  outData = np.load(outFile, allow_pickle=True)
  portV = outData["portV"]
  qubitV = outData["qubitV"]
  savedPsi = outData["savedPsi"]
  diagnostics = outData["diagnostics"].item()
  # res = np.load(resFile)
  # os.remove(csrFile)
  # os.remove(outFile)
  # os.remove(resFile)

  savedField = []
  # savedField, savedPsi, diagnostics = [[], [], []]
  return portV, qubitV, savedField, savedPsi, diagnostics

def solveTD_SciPy(matStruct, connLists, basisMaps, probP, solveP, outP, simType):
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

  dt, numT, beta, Vs = probP["dt"], probP["numT"], probP["beta"], probP["driveP"]["Vs"]
  basisOrder, quadOrder3D = probP["bOrder"], probP["3DIntOrder"]

  ep0 = 1e-9 / (36 * np.pi)
  mu0 = 4e-7 * np.pi

  numVec = K1.shape[0] # Number of vector basis functions

  print("Setting up time marching matrices")
  start_time = time.time()
  if simType == "E":
    if beta == 0: # Central differencing
      advMat_full = K2/dt**2 + (Ree+Bee)/(2*dt)
      diagVec = 1/np.sqrt(np.abs(advMat_full.diagonal()))
      M_diag = spa.diags_array(diagVec, format="csr")
      M_diag_inv = spa.diags_array(np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
      advMat_full = M_diag @ advMat_full @ M_diag
      def advMat(v):
        v = diagVec * v
        return diagVec * ((K2@v)/dt**2 + (Ree@v+Bee@v)/(2*dt)) # Matrix for the advanced (unknown) timestep
      def curMat(v):
        return diagVec * (2*(K2@v)/dt**2 - (K1@v)) # Matrix for the current (known) timestep
      def preMat(v):
        return diagVec * (-(K2@v)/dt**2 + (Ree@v+Bee@v)/(2*dt)) # Matrix for the prior (known) timestep
      
    else:
      advMat_full = K2/dt**2 + (Ree+Bee)/(2*dt) + beta*K1
      M_diag = spa.diags_array(1/np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
      M_diag_inv = spa.diags_array(np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
      advMat_full = M_diag @ advMat_full @ M_diag
      f = M_diag @ f
      def advMat(v):
        v = M_diag @ v
        return M_diag @ ((K2@v)/dt**2 + ((Ree+Bee)@v)/(2*dt) + beta*(K1@v)) # Matrix for the advanced (unknown) timestep
      def curMat(v):
        return M_diag @ (2*(K2@v)/dt**2 - (1-2*beta)*(K1@v)) # Matrix for the current (known) timestep
      def preMat(v):
        return M_diag @ (-(K2@v)/dt**2 + ((Ree+Bee)@v)/(2*dt) - beta*(K1@v)) # Matrix for the prior (known) timestep
      
  elif simType == "A":
    # Renormalize to adjust matrix condition number
    K1_coeff = mu0
    Ken_coeff = 1/ep0
    Kne_coeff = mu0*ep0
    Gnn_coeff = 1
    advMat_full = spa.block_array([[K1_coeff*(K2/dt**2 + (Ree+Bee)/(2*dt) + beta*K1), -Ken_coeff*(Ken/(2*dt) + beta*Ren)], 
                                        [Kne_coeff*Kne/(2*dt), -Gnn_coeff*(Gnn/(dt**2) + Bnn/(2*dt))]])  
    M_diag = spa.diags_array(1/np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
    M_diag_inv = spa.diags_array(np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
    advMat_full = M_diag @ advMat_full @ M_diag
    f = M_diag @ f
    def advMat(v):
      v = M_diag @ v
      return M_diag @ np.concatenate((K1_coeff*((K2@v[:numVec])/dt**2 + ((Ree+Bee)@v[:numVec])/(2*dt) + beta*K1@v[:numVec]) \
                                      - Ken_coeff*(Ken/(2*dt) + beta*Ren)@v[numVec:], \
                                      Kne_coeff*Kne@v[:numVec]/(2*dt) - Gnn_coeff*(Gnn/(dt**2) + Bnn/(2*dt))@v[numVec:]))
    def curMat(v):
      return M_diag @ np.concatenate((2*(K2@v[:numVec])/dt**2 - (1-2*beta)*K1@v[:numVec] \
                                      + ((1-2*beta)*Ren)@v[numVec:], \
                                      2*(Gnn@v[numVec:])/(dt**2)))
    def preMat(v):
      return M_diag @ np.concatenate((-(K2@v[:numVec])/dt**2 + ((Ree+Bee)@v[:numVec])/(2*dt) - beta*K1@v[:numVec] \
                                      - (Ken@v[numVec:])/(2*dt) + beta*(Ren@v[numVec:]), \
                                      - (Kne@v[:numVec])/(2*dt) + (Gnn@v[numVec:])/(dt**2) - (Bnn@v[numVec:])/(2*dt)))
    
  elif simType == "TCS":
    advMat_full = spa.block_array([[(K2/dt**2 + (Ree+Bee)/(2*dt) + beta*K1), Ken/dt**2 + (Ren+Ben)/(2*dt)], 
                                        [Kne/dt**2 + (Rne+Bne)/(2*dt), Gnn/dt**2 + (Rnn+Bnn)/(2*dt)]])  
    M_diag = spa.diags_array(1/np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
    M_diag_inv = spa.diags_array(np.sqrt(np.abs(advMat_full.diagonal())), format="csr")
    advMat_full = M_diag @ advMat_full @ M_diag
    f = M_diag @ f
    def advMat(v):
      v = M_diag @ v
      return M_diag @ np.concatenate(((K2@v[:numVec])/dt**2 + (Ree@v[:numVec]+Bee@v[:numVec])/(2*dt) + beta*K1@v[:numVec] \
                                      + (Ken@v[numVec:])/dt**2 + (Ren@v[numVec:]+Ben@v[numVec:])/(2*dt), \
                                      (Kne@v[:numVec])/dt**2 + (Rne@v[:numVec]+Bne@v[:numVec])/(2*dt) \
                                      + (Gnn@v[numVec:])/dt**2 + (Rnn@v[numVec:]+Bnn@v[numVec:])/(2*dt)))
    def curMat(v):
      return M_diag @ np.concatenate(((2*K2@v[:numVec])/dt**2 - (1-2*beta)*K1@v[:numVec] \
                                      + (2*Ken@v[numVec:])/dt**2, \
                                      (2*Kne@v[:numVec])/dt**2 \
                                      + (2*Gnn@v[numVec:])/dt**2))
    def preMat(v):
      return M_diag @ np.concatenate(((-K2@v[:numVec])/dt**2 + (Ree@v[:numVec]+Bee@v[:numVec])/(2*dt) - beta*K1@v[:numVec] \
                                      + (-Ken@v[numVec:])/dt**2 + (Ren@v[numVec:]+Ben@v[numVec:])/(2*dt), \
                                      (-Kne@v[:numVec])/dt**2 + (Rne@v[:numVec]+Bne@v[:numVec])/(2*dt) \
                                      + (-Gnn@v[numVec:])/dt**2 + (Rnn@v[numVec:]+Bnn@v[numVec:])/(2*dt)))
    

  advMatOp = spa.linalg.LinearOperator(advMat_full.shape, matvec=advMat)
  print(f"Time marching matrices constructed. Time elapsed: {(time.time() - start_time):.3f} s")

  start_time = time.time()
  advMat_ilu = spa.linalg.spilu(advMat_full.tocsc(), drop_tol=solveP["dropTol"], fill_factor=solveP["rowFill"])
  M = spa.linalg.LinearOperator(advMat_full.shape, advMat_ilu.solve)
  M_time = time.time() - start_time
  M_sparsity = 100*(1-(advMat_ilu.L.nnz + advMat_ilu.U.nnz)/(advMat_full.shape[0]**2))
  print(f"Preconditioning done. Time elapsed: {M_time:.3f} s. Sparsity percentage: {M_sparsity:.3f}%") 

  diagnostics = {}
  if solveP["saveInfo"]:
    diagnostics["M_time"] = M_time
    diagnostics["M_sparsity"] = M_sparsity
    diagnostics["rowFill"] = (solveP["rowFill"]/100)*advMat_full.shape[0]**2/advMat_full.nnz
    diagnostics["iter"] = np.zeros((numT,))
    diagnostics["dropTol"] = solveP["dropTol"]


  # Initialize matrices for storage
  V_ports = np.zeros((connLists["portP"]["num"], numT)) # Voltages over each port, saved for each timestep
  solVals = np.zeros((advMat_full.shape[0],3), dtype=np.float64)

  if outP["saveFields"] and len(outP["saveSteps"]) != 0: # Check for timesteps to save the electric field at
    if outP["saveEle"] == "all":
      savedField = np.zeros((len(connLists["tetList"])+len(connLists["wedgeList"]), 3, len(outP["saveSteps"]))) # For storing the electric field at those timesteps
    else:
      savedField = np.zeros((len(outP["saveEle"]), 3, len(outP["saveSteps"]))) # For storing the electric field at those timesteps
    saveInd = 0 # Tracks the current index of sp["saveField"] to check for
  else:
    savedField = []

  print("Starting time-marching")
  start_time = time.time()

  ## Start by getting things going in the second timestep
  if simType == "E" or simType == "TCS":
    b = -f*(Vs[2] - Vs[0])/(2*dt)
  elif simType == "A":
    b = f*(beta*Vs[2] + (1-2*beta)*Vs[1] + beta*Vs[0])

  if solveP["saveInfo"]:
    solVals[:,1], diagnostics["iter"][1] = itSolve_par(advMatOp, b, 1/dt, solveP["tol"], solveP["restart"], solveP["maxit"], M=M, saveDiagnostics=True)
    if diagnostics["iter"][1] == solveP["restart"]*solveP["maxit"]:
      print(f"Iterative solver failed to converge on timestep {1}. Terminating time-marching.")
      V_ports[:,1:] = -1
      return V_ports, savedField, diagnostics
  else:
    solVals[:,1], _ = itSolve_par(advMatOp, b, 1/dt, solveP["tol"], solveP["restart"], solveP["maxit"], M=M)
  solVals[:,1] = M_diag @ solVals[:,1]
  if simType == "A":
    solVals[:numVec,1] = solVals[:numVec,1]*K1_coeff
    solVals[numVec:,1] = solVals[numVec:,1]*Ken_coeff

  # Evaluate voltage for this timestep
  V_ports[:,1] = getPortVoltages(solVals[:,:2], basisMaps["portVoltVecInds"], basisMaps["portVoltVecVals"], connLists["portP"], K1.shape[0], simType, "TD", dt=dt)

  ## Now iterate through the remaining timesteps
  for ii in range(2, numT-1):
    if simType == "E" or simType == "TCS":
      b = -f*(Vs[ii+1] - Vs[ii-1])/(2*dt)  + curMat(solVals[:,1]) + preMat(solVals[:,0])
    elif simType == "A":
      # b = np.concatenate((f*(beta*Vs[ii+1] + (1-2*beta)*Vs[ii] + beta*Vs[ii-1]), np.zeros((N_n,)))) \
      #                 + curMat(solVals[:,1]) + preMat(solVals[:,0]) + recurMat(cumSolVals)
      b = f*(beta*Vs[ii+1] + (1-2*beta)*Vs[ii] + beta*Vs[ii-1]) \
            + curMat(solVals[:,1]) + preMat(solVals[:,0])
      b[numVec:] /= Ken_coeff ## TODO: Double check that this scaling is correct

    guessVec = M_diag_inv @ solVals[:,1] # Rescale the initial guess
    if simType == "A":
      guessVec[:numVec] /= K1_coeff
      guessVec[numVec:] /= Ken_coeff

    if solveP["saveInfo"]:
      solVals[:,2], diagnostics["iter"][ii] = itSolve_par(advMatOp, b, 1/dt, solveP["tol"], solveP["restart"], solveP["maxit"], M=M, x0=guessVec, saveDiagnostics=True, debug=True)
      if diagnostics["iter"][ii] == solveP["restart"]*solveP["maxit"]:
        print(f"Iterative solver failed to converge on timestep {ii}. Terminating time-marching.")
        V_ports[:,ii:] = -1
        return V_ports, savedField, diagnostics
    else:
      solVals[:,2], _ = itSolve_par(advMatOp, b, 1/dt, solveP["tol"], solveP["restart"], solveP["maxit"], M=M, x0=guessVec)
    solVals[:,2] = M_diag @ solVals[:,2]

    if simType == "A":
      solVals[:numVec,2] = solVals[:numVec,2]*K1_coeff
      solVals[numVec:,2] = solVals[numVec:,2]*Ken_coeff

    # Evaluate port voltages for this timestep
    V_ports[:,ii] = getPortVoltages(solVals[:,1:], basisMaps["portVoltVecInds"], basisMaps["portVoltVecVals"], connLists["portP"], K1.shape[0], simType, "TD", dt=dt)

    # Check whether field needs to be saved - if yes, calculate it
    if outP["saveFields"] and ii == outP["saveSteps"][saveInd]:
      save_time = time.time()
      savedField[:,:,saveInd] = getTotField(connLists, basisMaps, simType, quadOrder3D, basisOrder, solVals, "TD", outP["eleToSave"], dt=dt)
      if saveInd != len(outP["saveSteps"]) - 1:
        saveInd += 1
      print(f"Time spent saving the field: {(time.time() - save_time):.3f} s")

    # Rotate the solution vectors to prep for the next timestep
    solVals = np.concatenate((solVals[:,1:], np.zeros((advMat_full.shape[0],1))), axis=1)
    
  solve_time = time.time() - start_time
  print(f"Time marching done. Time elapsed: {solve_time:.3f} s")
  if solveP["saveInfo"]:
    diagnostics["solveTime"] = solve_time

  return V_ports, savedField, diagnostics


def solveTD_PETSc(matStruct, subDLists, connLists, basisMaps, probP, solveP, outP, simType, numProcs):
  if "vecSubCutoffs" in subDLists:
    vecSubCutoffs, vecSubDs = subDLists["vecSubCutoffs"], subDLists["vecSubDs"]
    scalSubCutoffs, scalSubDs = subDLists["scalSubCutoffs"], subDLists["scalSubDs"]
  else:
    vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs = [[], [], [], []]

  csrFile = "csrData_TD.npz"
  outFile = "PETSc_result_TD.npy"
  resFile = "PETSc_residual.npy"
  saveCSRData(matStruct, [], numProcs, csrFile, vecSubCutoffs, vecSubDs, scalSubCutoffs, scalSubDs, \
              Vs=probP["driveP"]["Vs"], portVoltVecInds=basisMaps["portVoltVecInds"], portVoltVecVals=basisMaps["portVoltVecVals"])

  cmd = buildPETScCmd("PETSc_time_march.py", solveP, matStruct, csrFile, simType, vecSubCutoffs, numProcs)
  dt, numT, beta = [probP["dt"], probP["numT"], probP["beta"]]
  cmd.extend(["-dt", f"{dt}", "-numT", f"{numT}", "-beta", f"{beta}"])
  print(cmd)

  # Solve the problem in PETSc
  result = subprocess.run(cmd, capture_output=True, text=True)
  print(result.stdout)
  print(result.stderr)

  # Load solution and residual from files, delete files and return
  portV = np.load(outFile)
  # res = np.load(resFile)
  # os.remove(csrFile)
  # os.remove(outFile)
  # os.remove(resFile)

  return portV, [], []
