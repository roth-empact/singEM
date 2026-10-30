import os
import subprocess
import sys
import shutil
import time
import numpy as np
import json
from optimizeBO import delete_folder
def generate_geometries(points,meshDir,runPara,simType,simGeo,simConfig,resultDir,numProcs,Code):
	if os.path.exists(meshDir):
		delete_folder(meshDir)
	with open("./Bayesian_Optimization/Files/meshGenerate.txt","w") as f:
		for i,p in enumerate(points):
			PadX,PadY,JuncLen,Pad2GndY,Pad2GndX,TrenchZ,TaperW,Kval=p.tolist()
			print(f"PadX={PadX}, PadY={PadY}, JuncLen={JuncLen}, Pad2GndY={Pad2GndY},Pad2GndX={Pad2GndX},TrenchZ={TrenchZ},TaperW={TaperW}, and Kval={Kval} which is point number {i}",flush=True)
			f.write(f"python {Code} {PadX} {PadY} {JuncLen} {Pad2GndY} {Pad2GndX} {TrenchZ} {TaperW} {Kval} {meshDir} {i}")
			f.write("\n")
	if simType == "Serial":
		## Serial Mesh Generation
		start=time.time()
		with open("./Bayesian_Optimization/Files/meshGenerate.txt", "r") as f:
			for line in f:
				command = line.strip()
				if command:
					subprocess.run(command.split(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, check=True)
		print(f"Serial mesh generation took {time.time()-start:.4f} seconds",flush=True)
		
	else:
		## Parallel Mesh Generation --- needs Hypershell
		start=time.time()
		## Update task list
		subprocess.run(
		["hs","submit","./Bayesian_Optimization/Files/meshGenerate.txt"]
		)
		##
		subprocess.run(
		["hsx",f"-N{runPara}","--capture"],check=True
		)
		print(f"Parallel mesh generation took {time.time()-start:.4f} seconds",flush=True)
	fileParaOrganize(simGeo,simConfig,meshDir,resultDir,numProcs)
	
def fileParaOrganize(simGeo,simConfig,meshDir,resultDir,numProcs):
	Mesh_files=[f for f in os.listdir(meshDir) if f.endswith(".msh")]

	with open(simConfig,"r") as f:
		data=json.load(f)

	Json_list=[]
	Result_list=[]


	for ii in Mesh_files:
		data["probP"]["meshFile"]=meshDir+ii
		data["probP"]["geomFile"]=simGeo
		data["numProcs"]=numProcs
		data["outP"]["resFile"]=resultDir+"/"+ii[:-4]+"_result.npz"
		Result_list.append(resultDir+"/"+ii[:-4]+"_result.npz")
		with open(meshDir+ii[:-4]+"_config.json","w") as f:
			Json_list.append(meshDir+ii[:-4]+"_config.json")
			json.dump(data,f,indent=4)
	with open("./Bayesian_Optimization/Files/runGeoMesh.txt","w") as f:
		for ii in Json_list:
			f.write(f"python main.py -f {ii}")
			f.write("\n")
	with open("./Bayesian_Optimization/Files/runResults.txt","w") as f:
		for ii in Result_list:
			f.write(f"{ii}")
			f.write("\n")
			
