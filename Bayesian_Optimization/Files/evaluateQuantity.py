import subprocess
import numpy as np
import re
import time
import torch
import os
import shutil

def nearby_points(x, percent=0.05, n_points=8, lower=None, upper=None,param_range=None):
	d=x.numel()
	points = []
	scale = percent * torch.abs(x)
	scale=torch.maximum(scale,torch.full_like(scale,1e-8))
	while len(points)<n_points:
		direction = torch.randn(d,device=x.device,dtype=x.dtype)
		direction /= direction.norm()
		candidate = x + scale * direction
		if lower is not None:
			if torch.any(candidate < lower):
				continue
		if upper is not None:
			if torch.any(candidate > upper):
				continue
		points.append(candidate)
	return torch.stack(points)




def make_predict_Q(Q_model):
	
	def predict_Q(x):
		x=torch.tensor(x,dtype=torch.double).reshape(1,-1)
		with torch.no_grad():
			return Q_model.posterior(x).mean.item()
	return predict_Q

def make_predict_C(C_model):
	
	def predict_C(x):
		x=torch.tensor(x,dtype=torch.double).reshape(1,-1)
		with torch.no_grad():
			return C_model.posterior(x).mean.item()
	return predict_C

def make_predict_Qstd(Q_model):
	
	def predict_Qstd(x):
		x=torch.tensor(x,dtype=torch.double).reshape(1,-1)
		with torch.no_grad():
			return Q_model.posterior(x).variance.sqrt().item()
	return predict_Qstd

def make_predict_Cstd(C_model):
	
	def predict_Cstd(x):
		x=torch.tensor(x,dtype=torch.double).reshape(1,-1)
		with torch.no_grad():
			return C_model.posterior(x).variance.sqrt().item()
	return predict_Cstd

def make_capacitance_constraintU(C_target,C_tol):
	
	def capacitance_constraint(Y,X=None):
		C = Y[...,1]
		#print(f"Upper Constraint with shape {C.shape}",flush=True)
		return C-C_target-C_tol
	
	return capacitance_constraint
	
def make_capacitance_constraintL(C_target,C_tol):
	
	def capacitance_constraint(Y,X=None):
		C = Y[...,1]
		#print(f"Lower Constraint with shape {C.shape}",flush=True)
		return C_target-C-C_tol
	
	return capacitance_constraint


def cost_eval(batch_size,Ntrain_x,tanD,directory_O):
	dir_Failed = "./Bayesian_Optimization/Failed"
	dir_Optim = directory_O
	with open("./Bayesian_Optimization/Files/runResults.txt","r") as f:
		Res=f.read().splitlines()
		
	tanD=np.array(tanD)
	len_Big_Data=len(tanD)+1
	Big_Data=np.zeros((len_Big_Data,batch_size))
	Quality=np.zeros(batch_size)
	Capacitance=np.zeros(batch_size)
	Capacitance2=np.zeros(batch_size)
	design_fail=False
	
	for ii in range(len(Res)):
		val=int(re.search(r"test_(\d+)_result\.npz$",Res[ii]).group(1))
		
		if os.path.exists(Res[ii]):
			data=np.load(Res[ii])
		else:
			design_fail=True
			for filename in os.listdir(dir_Optim):
				if os.path.basename(Res[ii])[:-11] in filename:
					shutil.move(os.path.join(dir_Optim,filename),os.path.join(dir_Failed,filename))
					print(f" FAILURE!!! {filename} moved",flush=True)
			continue
		if len(data['freqs'])>1:
			surfPart=np.real(data['outInts'][-1].flatten())
			Cap=np.abs(-1/(2*np.pi*data['freqs'][0]*np.imag(data['portV'][0])))
			Cap2=np.abs(-1/(2*np.pi*data['freqs'][-1]*np.imag(data['portV'][-1])))
		else:
			surfPart=np.real(data['outInts'][0].flatten())
			Cap=np.abs(-1/(2*np.pi*data['freqs'][0]*np.imag(data['portV'][0])))
			Cap2=[0]
		Big_Data[0:(len_Big_Data-1),val]=surfPart[[0,1,2,6,7,8,9]]
		Big_Data[len_Big_Data-1,val]=1/np.sum(Big_Data[:-1,val]*tanD)
		Quality[val]=Big_Data[len_Big_Data-1,val]
		Capacitance[val]=Cap[0]*1e15
		Capacitance2[val]=Cap2[0]*1e15
	if any(Capacitance2):
		print(f"Capacitance at {data['freqs'][0]*1e-9} GHz = {Capacitance}")
		print(f"Capacitance at {data['freqs'][-1]*1e-9} GHz = {Capacitance2}")
		print(f"Percentage difference {((Capacitance)-(Capacitance2))/(Capacitance)*100}")
	if design_fail:
		mask_0=Quality !=0
		Quality=Quality[mask_0]
		Capacitance=Capacitance[mask_0]
		Ntrain_x=Ntrain_x[mask_0]
		Big_Data=Big_Data[:,mask_0]
		
	return Quality,Capacitance,Ntrain_x,Big_Data
		
		

def run_simulations(parallel_cur,batch_size,Ntrain_x,simType,tanD,directory_O):
	start=time.time()
	if simType == "Serial":
		start=time.time()
		with open("./Bayesian_Optimization/Files/runGeoMesh.txt", "r") as f:
			for line in f:
				command = line.strip()
				if command:
					subprocess.run(command.split(), check=True)
		print(f"Serial mesh generation took {time.time()-start:.4f} seconds",flush=True)
		return cost_eval(batch_size,Ntrain_x,tanD,directory_O)

	else:
		subprocess.run(
		["hs","submit","./Bayesian_Optimization/Files/runGeoMesh.txt"]
		)
		## Check how many loaded
		count=subprocess.run(
		["hs","list","--remaining","--count"],
		capture_output=True,
		text=True
		)
		print(f"Parallel simulation starting, number of loaded tasks: {count.stdout.strip()}")
		##
		subprocess.run(
		["hsx",f"-N{parallel_cur}","--capture"],check=True
		)
		end=time.time()
		print(f"Parallel simulation took {end-start:.4f} seconds")
		return cost_eval(batch_size,Ntrain_x,tanD,directory_O)
	
