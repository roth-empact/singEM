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


import os
import numpy as np
import sys
import time
sys.path.append('/PV_visualization/')
from PV_visualization.pvController import *
from mesh import initMesh
from meshToMats import meshToMats, buildVolMats
from solveFD import solveFD
from solveTD import solveTD, solveMS
from solveEig import solveEig
import postProcess as post
import argparse
from JSON_util import readSimJSON
from ReadExodusMesh import writeExResults
from linalg_funcs import modify_f_vector
from gmshToExo import gmshToExo

def main(simFile, verbose):
  """
  Overall wrapper for finite element simulations. Supply with a JSON file
  for configuring the simulation, which can be generated with writeSimJSON() in JSON_util.py.
  That file must reference a mesh file generated with gmsh or Coreform Cubit, 
  as well as a geometry configuration file generated with writeGeometryJSON() in JSON_util.py.

  Inputs:
    simFile - Path to JSON configuration file containing simulation parameters 
              and references to the geometry and mesh files
    verbose - Indicates whether to provide additional printout information

  """
  begin_time = time.time()

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
      case "Phi":
        methodStr = "Poisson"
    match probP["type"]:
      case "FD":
        probStr = "frequency domain"
      case "TD":
        probStr = "time domain"
      case "Eig":
        probStr = "eigenmode"
      case "MS":
        probStr = "Maxwell-Schroedinger"
    geomFile = probP["geomFile"]
    print(f"Using simulation settings from {simFile} and geometry settings from {geomFile}")
    meshFile = probP["meshFile"]
    print(f"Solving {probStr} problem on {meshFile} using {methodStr} with {numProcs} processors.")

  # Prepare filenames for AMR (if used)
  curResFile = outP["resFile"]
        
  for passNum in range(probP["maxAMR"]):
    pass_start = time.time()
    ## Produce connectivity lists from the mesh and geometry files
    connLists, meshStats, subDLists, singEleLists = initMesh(probP["meshFile"], probP["geomFile"], simType, probP["numSubDs"], probP["outBC"])
    resDict = {"meshTime": time.time() - pass_start, "meshStats": meshStats}
    N_n, N_e, N_w_e, N_f, N_q_f, N_tet, N_wedge = \
      meshStats["N_n"], meshStats["N_e"], meshStats["N_w_e"], meshStats["N_f"], meshStats["N_q_f"], meshStats["N_tet"], meshStats["N_wedge"]
    if verbose:
      print(f"Time spent forming element lists: {(time.time() - pass_start):.3f} s")
      print(f"Number of tetrahdrons: {N_tet}")
      print(f"Number of wedges: {N_wedge}")
      print(f"Number of nodes: {N_n}")
      print(f"Number of edges: {N_e+N_w_e}")
      print(f"Number of triangular faces: {N_f}")
      print(f"Number of quadrilateral faces: {N_q_f}")

      l_max, l_min, l_avg, l_stdev = meshStats["l_max"], meshStats["l_min"], meshStats["l_avg"], meshStats["l_stdev"]
      print(f"Longest triangular edge: {(l_max*1e3):.3e} mm")
      print(f"Shortest triangular edge: {(l_min*1e3):.3e} mm")
      print(f"Average triangular edge length: {(l_avg*1e3):.3e} mm")
      print(f"Triangular edge length standard deviation: {(l_stdev*1e3):.3e} mm")

    # Mark whether singular functions are being used in the solver parameters (for determining LU memory constraints)
    if len(singEleLists["singTets"]) != 0 or len(singEleLists["singWedges"]) != 0:
      solveP["singFlag"] = True
    else:
      solveP["singFlag"] = False

    ## Form matrices and driving vectors
    start_time = time.time()
    matStruct, basisMaps, subDLists \
      = meshToMats(connLists, subDLists, singEleLists, probP["bOrder"], probP["sOrder"], probP["bOrder_long"], probP["sOrder_long"], \
                   probP["singEdgeFlag"], probP["3DIntOrder"], probP["2DIntOrder"], simType, probP["outBC"], meshStats, numProcs)
    resDict["unknownCnt"] = len(matStruct["f"])
    resDict["matTime"] = time.time() - start_time

    ## Solve the problem using the matrices and process the outputs
    start_time = time.time()
    match probP["type"]:
      case "FD":
        freqs = probP["freqs"]
        resDict["freqs"] = freqs
        
        if probP.get("portMatrix", False):
          numPorts = connLists["portP"]["num"]
          if numPorts > 0:
            print(f"portMatrix mode: {numPorts} ports")
            
            portV_all = np.zeros((numPorts, numPorts, len(freqs)), dtype=np.complex128)
            x_ref = np.zeros((len(matStruct["f"]), numPorts), dtype=np.complex128) # Solutions at the reference frequency for AMR 
            
            for port_idx in range(numPorts):
              print(f"Solving port {port_idx+1}/{numPorts}...")
              
              # Update the source vector and solve the problem
              matStruct = modify_f_vector(matStruct, port_idx, basisMaps)
              x, _ = solveFD(matStruct, subDLists, freqs, solveP, simType, numProcs, outP["resFile"])
              portV = post.getPortVoltages(x, basisMaps["portVoltVecInds"], basisMaps["portVoltVecVals"], \
                                           basisMaps["portVoltScalInds"], basisMaps["portVoltScalVals"], \
                                           matStruct["K1"].shape[0], connLists["portP"], simType, "FD", freqs=freqs)
              qubitV = post.getPortVoltages(x, basisMaps["qubitVoltVecInds"], basisMaps["qubitVoltVecVals"], \
                                            basisMaps["qubitVoltScalInds"], basisMaps["qubitVoltScalVals"], matStruct["K1"].shape[0], \
                                            connLists["qubitP"], simType, "FD", freqs=freqs)
              
              # Save solution for AMR
              if 1: # TODO: Add option for dynamic selection of refinement index
                # Select a port other than the driving one to determine the reference frequency from
                if port_idx == 1:
                  refPort = 0
                else:
                  refPort = 1
                probP["refFreqInd"] = np.argmax(np.abs(portV[:,refPort]))
              x_ref[:,port_idx] = x[:,probP["refFreqInd"]]

              portV_all[:, port_idx, :] = portV.T
            
            # Calculate output surface integrals for portMatrix mode
            if connLists["outP"]["num"] > 0:
              resDict["outInts"] = post.getOutInts_tets_loop(x, connLists, basisMaps, singEleLists, connLists["outP"], \
                                                             probP["2DIntOrder"], probP["bOrder"], probP["sOrder"], probP["bOrder_long"], probP["sOrder_long"], meshStats["N_tet"], simType, \
                                                             matStruct["K1"].shape[0], "FD", numProcs, freqs=freqs)
              
            sMats = np.zeros((numPorts, numPorts, len(freqs)), dtype=np.complex128)
            for ii in range(len(freqs)):
              sMats[:,:,ii] = post.zMatToS(portV_all[:,:,ii])
              # sMats[:,:,ii] = post.getSParams(connLists["portP"], portV_all[:,:,ii])
            
            if outP["saveV"]:
              resDict["portV"] = portV_all
            if outP["saveS"]:
              resDict["sMats"] = sMats
            
            if verbose:
              print(f"\nSimulation Results:")
              for ii in range(len(freqs)):
                print(f"Current frequency: {(freqs[ii]/1e9):.5f} GHz")
                for portNum in range(numPorts):
                  for excNum in range(numPorts):
                    print(f"Voltage at port {portNum} (exc {excNum}): {portV_all[portNum, excNum, ii]:.5f} V")
                if outP["saveS"]:
                  print(f"S11: {20*np.log10(np.abs(sMats[0,0,ii])):.5f} dB, S21: {20*np.log10(np.abs(sMats[1,0,ii])):.5f} dB")
                  print(f"S12: {20*np.log10(np.abs(sMats[0,1,ii])):.5f} dB, S22: {20*np.log10(np.abs(sMats[1,1,ii])):.5f} dB")
          else:
            print("portMatrix mode enabled but no ports found!")
            
        else:
          # Normal mode: single excitation
          x, diagnostics = solveFD(matStruct, subDLists, freqs, solveP, simType, numProcs, outP["resFile"])
          resDict["diagnostics"] = diagnostics
          
          if verbose:
            print(f"Time spent solving the FD system: {(time.time()-start_time):.3f} s.")
          
          if connLists["outP"]["num"] > 0:
            start_time = time.time()
            resDict["outInts"] = post.getOutInts_tets_loop(x, connLists, matStruct, basisMaps, singEleLists, connLists["outP"], \
                                                           probP["2DIntOrder"], probP["bOrder"], probP["sOrder"], probP["bOrder_long"], probP["sOrder_long"], meshStats["N_tet"], simType, \
                                                           matStruct["K1"].shape[0], "FD", numProcs, freqs=freqs)
            print(f"Time spent calculating output surface integrals: {(time.time()-start_time):.3f} s.")

          if outP["saveV"] or outP["saveS"]:
            if simType == "E" or simType == "A" or simType == "TCS":
              portV = post.getPortVoltages(x, basisMaps["portVoltVecInds"], basisMaps["portVoltVecVals"], basisMaps["portVoltScalInds"], basisMaps["portVoltScalVals"], matStruct["K1"].shape[0], \
                                          connLists["portP"], simType, "FD", freqs=freqs)
              qubitV = post.getPortVoltages(x, basisMaps["qubitVoltVecInds"], basisMaps["qubitVoltVecVals"], basisMaps["qubitVoltScalInds"], basisMaps["qubitVoltScalVals"], matStruct["K1"].shape[0],\
                                            connLists["qubitP"], simType, "FD", freqs=freqs)
              # Calculate current along voltage sources
              voltageI = post.getSurfaceCurrents(x, freqs, basisMaps["voltCurVecVals"], basisMaps["voltCurVecInds"], simType)
              resDict["portV"], resDict["qubitV"], resDict["voltageI"] = [portV, qubitV, voltageI]

              if outP["saveS"]:
                portS = post.getSParams(connLists["portP"], resDict["portV"])
                resDict["portS"] = portS
            elif simType == "Phi":
              # Calculate charge along voltage sources and ports
              qVec = np.real(matStruct["Tnn"] @ x[:,0]) # Vector of charge densities at each node
              portQ = post.getPortCharges(qVec, basisMaps["scalNodeMap"], connLists["nodeList"], connLists["portList"], connLists["portP"])
              voltQ = post.getPortCharges(qVec, basisMaps["scalNodeMap"], connLists["nodeList"], connLists["voltList"], connLists["voltP"])
              resDict["portQ"], resDict["voltQ"] = [portQ, voltQ]

          # Save solution for AMR
          if passNum != probP["maxAMR"]-1: # TODO: Add option for dynamic selection of refinement index
            probP["refFreqInd"] = np.argmax(np.abs(portV[:,connLists["portP"]["num"]-1]))
            x_ref = x[:,probP["refFreqInd"]].reshape((x.shape[0], 1)) # Solutions at the reference frequency for AMR

          if verbose:
            print(f"\nResults for pass {passNum}:")
            if simType == "E" or simType == "A" or simType == "TCS":
              for ii in range(len(freqs)):
                print(f"Current frequency: {(freqs[ii]/1e9):.5f} GHz")
                for portNum in range(connLists["portP"]["num"]):
                  print(f"Voltage at port {portNum}: {portV[ii][portNum]:.5f} V")
                for qubitNum in range(connLists["qubitP"]["num"]):
                  print(f"Voltage at qubit {qubitNum}: {qubitV[ii][qubitNum]:.5f} V")
                for voltNum in range(connLists["voltP"]["num"]):
                  print(f"Current at voltage source {voltNum}: {voltageI[ii][voltNum]:.5e} A")
                if connLists["outP"]["num"] > 0:
                  for outNum in range(connLists["outP"]["num"]):
                    curFieldType = connLists["outP"][f"{outNum}"]["fieldType"]
                    curOutInt = resDict["outInts"][ii][outNum]
                    print(f"Output surface {outNum} {curFieldType} integral: {curOutInt:.5e}")
                if outP["saveS"]:
                  print(f"S11: {20*np.log10(np.abs(portS[ii,0]))} dB, S21: {20*np.log10(np.abs(portS[ii,1]))} dB")
              
            elif simType == "Phi":
              for portNum in range(connLists["portP"]["num"]):
                print(f"Charge at portage source {portNum}: {portQ[portNum]:.5e} C")
              for voltNum in range(connLists["voltP"]["num"]):
                print(f"Charge at voltage source {voltNum}: {voltQ[voltNum]:.5e} C")
            
      case "TD":
        portV, savedField, diagnostics = solveTD(matStruct, subDLists, connLists, basisMaps, probP, solveP, outP, simType, numProcs)
        resDict["t"] = probP["t"]
        resDict["Vs"] = probP["driveP"]["Vs"]
        resDict["portV"] = portV
        if len(qubitV) != 0:
          resDict["qubitV"] = qubitV
          resDict["savedPsi"] = savedPsi
        resDict["diagnostics"] = diagnostics
      case "Eig":
        numEig = probP["numEig"]
        start_time = time.time()
        eigVals, x = solveEig(matStruct, numEig, solveP, simType, numProcs)
        eigSort = np.argsort(eigVals)
        eigVals = eigVals[eigSort]
        x = x[:,eigSort]
        freqs = np.sqrt(eigVals)/(2*np.pi)
        if verbose:
          print(f"{numEig} eigenvalues found in {(time.time() - start_time):.3f} s.")
          for ii, freq in enumerate(freqs):
            print(f"Eigenvalue {ii}: {(freq/1e9):.5f} GHz")
        resDict["eigvals"], resDict["eigvecs"] = [freqs, x]
      case "MS":
        portV, qubitV, savedField, savedPsi, diagnostics = solveMS(matStruct, subDLists, connLists, basisMaps, probP, solveP, outP, simType, numProcs)
        resDict["t"] = probP["t"]
        resDict["Vs"] = probP["driveP"]["Vs"]
        resDict["savedPsi"] = savedPsi
        resDict["portV"] = portV
        resDict["qubitV"] = qubitV
        resDict["diagnostics"] = diagnostics

    # If desired, calculate field at provided points
    if len(outP["fieldPoints"]) != 0:
      outP["fieldPoints"] *= connLists["scale"]
      resDict["fieldPoints"] = outP["fieldPoints"]
      resDict["pointFields"] = np.zeros((len(freqs), len(outP["fieldPoints"]), 3), dtype=x.dtype)
      for ii in range(len(freqs)):
        resDict["pointFields"][ii], resDict["pointTets"] \
          = post.getPointFields(outP["fieldPoints"], connLists, basisMaps, singEleLists, simType,
                                probP, x[:,ii], 2*np.pi*freqs[ii], matStruct["K1"].shape[0], \
                                "FD", outP["fieldType"])
        
    # Check if any of the volumes are marked to be perform field measurements
    volInts = []
    for ind, volP in enumerate([connLists["tetVolP"], connLists["wedgeVolP"]]):
      for ii in range(volP["num"]):
        if "saveFieldInt" in volP[f"vol{ii}"] and volP[f"vol{ii}"]["saveFieldInt"]:
          if "fieldType" in volP[f"vol{ii}"]:
            fieldType = volP[f"vol{ii}"]["fieldType"]
          else:
            fieldType = "E" # If the type of field isn't specified, assume it's the electric field
          
          # Create a temporary list of volume parameters for evaluating the field in the current volume
          tempVolP = {"num": 1}
          tempVolP["vol0"] = volP[f"vol{ii}"]

          # Determine where to cut off the list of elements
          if ii != volP["num"] - 1:
            endInd = volP[f"vol{ii+1}"]["startInd"]
          else:
            endInd = len(connLists[f"{eleStrLower}List"])

          # Integrate the field over the volume
          eleStr = ["Tet", "Wedge"][ind] # Determine which type of element is being looked at
          eleStrLower = eleStr.lower()
          if probP["type"] == "FD" or probP["type"] == "Eig":
            curVolInts = np.zeros((len(freqs),), dtype=np.complex128)
            for jj, freq in enumerate(freqs):
              # _, curVolInts[jj] = post.getEleField(connLists[f"{eleStrLower}List"][:endInd], connLists[f"{eleStrLower}EdgeList"][:endInd], singEleLists[f"{eleStrLower}SingEdgeList"][:endInd], connLists[f"{eleStrLower}FaceList"][:endInd], \
              #                                         tempVolP, connLists["nodeList"], basisMaps, singEleLists, simType, eleStr, fieldType, \
              #                                         probP["3DIntOrder"], probP["bOrder"], probP["sOrder"], x, "FD", "all", omega=2*np.pi*freq)
              tempMatStruct = buildVolMats(connLists[f"{eleStrLower}List"][:endInd], connLists[f"{eleStrLower}EdgeList"][:endInd], \
                                           singEleLists[f"{eleStrLower}SingEdgeList"][:endInd], connLists[f"{eleStrLower}FaceList"][:endInd], \
                                           singEleLists, tempVolP, connLists["nodeList"], connLists["abcList"], basisMaps, \
                                           probP["bOrder"], probP["sOrder"], probP["bOrder_long"], probP["sOrder_long"], probP["singEdgeFlag"], probP["3DIntOrder"], simType, "Tet", probP["outBC"], numProcs)
              curVolInts[jj] = np.real(x[:,jj].conj().T @ tempMatStruct["K2"] @ x[:,jj]) / 2
          volInts.append(curVolInts)

    if len(volInts) != 0: # Save the volume integrals to a file (if there are any)
      resDict["volInts"] = np.array(volInts)
      if verbose:
        print(f"Volume integrals: {volInts}")
        
    # Save results to a file (if desired)
    if curResFile != None:
      resDict["totTime"] = time.time() - begin_time
      resDict["meshStats"] = meshStats
      np.savez(curResFile, **resDict)

  # Plot the fields (if desired)
  if outP["saveFields"]:
    # eleInd = 4#singEleLists["singLineTets"][0]
    # post.getSingleTetField(x[:,1], connLists["tetList"][eleInd], connLists["tetEdgeList"][eleInd], connLists["tetFaceList"][eleInd], \
    #                        connLists["nodeList"], basisMaps, singEleLists, eleInd, probP["bOrder"], simType, "E")
    numEle = len(connLists["tetList"])+len(connLists["wedgeList"])
    plotData = np.arange(numEle).reshape((numEle,1))
    normFactors = np.zeros((len(freqs),), dtype=np.float64)
    if probP["type"] == "FD" or probP["type"] == "Eig":
      numSaveSteps = np.size(freqs)
      for ii in range(numSaveSteps):
        fieldVals, normFactors[ii] = post.getTotField(connLists, basisMaps, singEleLists, simType, probP, meshStats["N_tet"], \
                                                      x[:,ii], "FD", outP["fieldType"], "all", omega=2*np.pi*freqs[ii])
        if outP["fieldType"] == "Phi":
          fieldVals = np.real(fieldVals) + np.imag(fieldVals)
          vecFlag = "False"
        else:
          fieldVals = np.real(fieldVals) + np.imag(fieldVals)
          vecFlag = "True"
        plotData = np.append(plotData, fieldVals, axis=1) # For scalar data (in y direction)
    elif probP["type"] == "TD" and outP["saveEle"] == "all":
      numSaveSteps = len(outP["saveSteps"])
      for ii in range(numSaveSteps):
        if outP["fieldType"] == "Mag":
          fieldVals = np.sqrt(np.sum(np.abs(savedField[ii]**2), axis=1)).reshape((numEle,1))
          vecFlag = "False"
        else:
          vecFlag = "True"
        plotData = np.append(plotData, fieldVals, axis=1)
      
  print(f'Total run time: {(time.time() - begin_time):.3f} s')
  if outP["saveFields"]:
    np.save(outP["fFile"], plotData)
    if(os.path.splitext(probP["meshFile"])[1] == '.e'):
      # If an exodus mesh file is used, just write data directly into it
      writeExResults(probP["meshFile"], plotData, outP["fieldType"])
      meshFile = probP["meshFile"]
    elif(os.path.splitext(probP["meshFile"])[1] == '.msh'):
      # If a gmsh file is used, convert to exodus and write data into that
      meshFile = os.path.splitext(probP["meshFile"])[0] + ".e"
      gmshToExo(probP["meshFile"], meshFile)
      writeExResults(meshFile, plotData, outP["fieldType"])
    elif(os.path.splitext(probP["meshFile"])[1] == '.inp'):
      # If an abaqus mesh file is used, supply it to the plotting tool
      meshFile = probP["meshFile"]

    launchPV([meshFile, outP["fFile"], f"{numSaveSteps}", vecFlag])

if __name__ == "__main__":
  #* Script Arguments *#
  parser = argparse.ArgumentParser(
                  prog='main.py',
                  description='Main wrapper for finite element simulations.')
  parser.add_argument('-f', '--simFile',
                      help=".json simulation file. See writeSimJSON() for format details.",
                      default='./Meshes/Cavities/ridgedCavity_Eig.json')
  parser.add_argument('-v', '--verbose', 
                      help="Prints details about simulation status to the command line.",
                      action="store_true")
  args = parser.parse_args()
    
  main(args.simFile, True)