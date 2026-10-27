import json
import os
import sys

geoJson = sys.argv[1]
configJson = sys.argv[2]
Mesh_dir = sys.argv[3]
Result_dir = sys.argv[4]
num_procs = sys.argv[5]

Mesh_files=[f for f in os.listdir(Mesh_dir) if f.endswith(".msh")]

with open(configJson,"r") as f:
	data=json.load(f)
	
with open(geoJson,"r") as f:
	geo=json.load(f)

Json_list=[]
Result_list=[]


for ii in Mesh_files:
	data["probP"]["meshFile"]=Mesh_dir+ii
	data["probP"]["geomFile"]=geoJson
	data["numProcs"]=num_procs
	data["outP"]["resFile"]=Result_dir+"/"+ii[:-4]+"_result.npz"
	Result_list.append(Result_dir+"/"+ii[:-4]+"_result.npz")
	with open(Json_dir+ii[:-4]+"_config.json","w") as f:
		Json_list.append(Json_dir+ii[:-4]+"_config.json")
		json.dump(data,f,indent=4)
with open("./Bayesian_Optimization/Files/runGeoMesh.txt","w") as f:
	for ii in Json_list:
		f.write(f"python main.py -f {ii}")
		f.write("\n")
with open("./Bayesian_Optimization/Files/runResults.txt","w") as f:
	for ii in Result_list:
		f.write(f"{ii}")
		f.write("\n")
			



