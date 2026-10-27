import torch
import os
import sys
sys.path.append("./Bayesian_Optimization/Files/")
import shutil
import numpy as np
import json
from botorch.models import SingleTaskGP,ModelListGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from scipy.stats import qmc
import time
from scipy.optimize import minimize
from generateGeometry import *
from evaluateQuantity import *


############### Helper Functions ##################
def objective(x):
    return -predict_Q(x)

def make_CC_L2(C_target,C_tol):
	
	def capacitance_constraint(x):
		return predict_C(x)-(C_target-C_tol)
	
	return capacitance_constraint

def make_CC_U2(C_target,C_tol):
	
	def capacitance_constraint(x):
		return (C_target+C_tol)-predict_C(x)
	
	return capacitance_constraint

def quality_constraint(Y,X=None):
	return Y[...,0]

def delete_folder(folder):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isdir(path) and not os.path.islink(path):
            delete_folder(path)
        else:
            os.remove(path)

def run_main(json_path):
	######## Json path ###############
	with open(json_path, "r") as f:
		data = json.load(f)

	directory_O = data["fileDirectory"]["meshFolder"]
	directory_R1 = data["fileDirectory"]["resultFolder"]
	directory_R2 = data["fileDirectory"]["gpFolder"]

	##### Make and clear folders for placing Designs########
	if os.path.exists(directory_O):
		delete_folder(directory_O)
	else:
		os.makedirs(directory_O, exist_ok=True)
	if os.path.exists(directory_R1):
		delete_folder(directory_R1)
	else:
		os.makedirs(directory_R1, exist_ok=True)
	if os.path.exists(directory_R2):
		delete_folder(directory_R2)
	else:
		os.makedirs(directory_R2, exist_ok=True)
	######### Design Variables ###########
	dtype=torch.double
	torch.set_default_dtype(dtype)

	paramNames = data["simBounds"]["paramNames"]
	lower = data["simBounds"]["paramLowerBound"]
	upper = data["simBounds"]["paramUpperBound"] 
	Pdict =  {
    name: {"index": i, "lower": lower[i], "upper": upper[i]}
    for i, name in enumerate(paramNames)
	}
	dim = data["simBounds"]["numParam"]
	bounds = torch.tensor([
		[Pdict[name]["lower"] for name in Pdict],
		[Pdict[name]["upper"] for name in Pdict]
	], dtype=dtype)

	########## Simulation Parameters ############
	n_initial = data["simSettings"]["numInitial"] ## Map out intial Space
	batch_size= data["simSettings"]["batchSize"]  ## batches for optimizing
	C_target = data["simBounds"]["capTarget"] # in fF
	C_tol = data["simBounds"]["capTol"] # in fF
	simType = data["simSettings"]["simType"]
	if simType == "Serial":
		mesh_parallel = batch_size
		parallel_cur = batch_size
	else:
		mesh_parallel = batch_size
		parallel_cur = data["simSettings"]["totalProcsParallel"] // data["simSettings"]["numProcsPerSim"]
	iterations = data["simSettings"]["numIter"]
	res_GK = 20 # Parameters for acquisition function optimization
	raw_GK = 2048 # Parameters for acquisition function optimization


	Old_bQ=0 ## Old best Quality
	Old_bX=0 ## Old best corresponding dimension
	Old_bQ_iter=0 ## iteration when best was found
	iteration=0 ## Starting iteration
	detectMeshSpike = data["simSettings"]["meshSpikeCheck"]
	########### Making Linear Constraints for X and Y dimensions ############
	lb = bounds[0,:]
	ub = bounds[1,:]
	portlen = 0.002*1e3 # Length of port in this example
	max_X = data["simBounds"]["limitX"]
	max_Y = data["simBounds"]["limitY"]
	x_ind=torch.tensor([Pdict["PadX"]["index"],Pdict["Junc_len"]["index"],Pdict["P2GndX"]["index"]],dtype=torch.long)
	x_coeff=torch.tensor([
		2*(ub[Pdict["PadX"]["index"]]-lb[Pdict["PadX"]["index"]]),
		2*(ub[Pdict["Junc_len"]["index"]]-lb[Pdict["Junc_len"]["index"]]),
		2*(ub[Pdict["P2GndX"]["index"]]-lb[Pdict["P2GndX"]["index"]])
	],dtype=dtype)
	x_rhs=max_X-(portlen)-2*lb[Pdict["PadX"]["index"]]-2*lb[Pdict["Junc_len"]["index"]]-2*lb[Pdict["P2GndX"]["index"]]

	y_ind=torch.tensor([Pdict["PadY"]["index"],Pdict["P2GndY"]["index"]],dtype=torch.long)
	y_coeff=torch.tensor([
		(ub[Pdict["PadY"]["index"]]-lb[Pdict["PadY"]["index"]]),
		2*(ub[Pdict["P2GndY"]["index"]]-lb[Pdict["P2GndY"]["index"]])
	],dtype=dtype)
	y_rhs=max_Y-lb[Pdict["PadY"]["index"]]-2*lb[Pdict["P2GndY"]["index"]]

	inequality_constraints=[
		(x_ind,-1*x_coeff,-1*x_rhs),
		(y_ind,-1*y_coeff,-1*y_rhs)
	]

	######################################################
	q_objective=GenericMCObjective(quality_constraint)
	## Constraint Capacitance
	cap_constraintU=make_capacitance_constraintU(C_target,C_tol)
	cap_constraintL=make_capacitance_constraintL(C_target,C_tol)

	##########Run initial random Design##########
	NcandidateX=torch.zeros(n_initial,dim,dtype=dtype)

	print("Getting Initial Sample",flush=True)
	print("Using Sobol Sequence",flush=True)
	sobol=SobolEngine(dimension=dim,scramble=True)
	Ntrain_x=sobol.draw(n_initial)
	mask=torch.ones(Ntrain_x.shape[0],dtype=torch.bool)
	for indices,coefficients,rhs in inequality_constraints:
		values = Ntrain_x[:,indices] @ coefficients
		mask &= values >= rhs-1e-8
	Ntrain_x = Ntrain_x[mask]
	print(f"Number of viable points from intial sample: {mask.sum().item()}")
	train_x=unnormalize(Ntrain_x,bounds)

	##########Run first set###############
	print(f"Performing iteration number 0",flush=True)
	first_time = time.time()
	generate_geometries(train_x,directory_O,mesh_parallel,simType,data["simSettings"]["simGeoJson"],data["simSettings"]["simConfigJson"],data["fileDirectory"]["resultFolder"],data["simSettings"]["numProcsPerSim"],data["simSettings"]["simMeshFile"])
	Q_val,C_val,Ntrain_x,Participation_Data=run_simulations(parallel_cur,n_initial,Ntrain_x,simType,data["simBounds"]["tanD"],directory_O) 
	print(f"Average Value of Capacitance of run designs {np.mean(C_val)}")
	print(f"Average Value of Quality of run designs {np.mean(Q_val)}")
	N_Q=torch.tensor(Q_val,dtype=dtype).reshape(-1,1)
	N_C=torch.tensor(C_val,dtype=dtype).reshape(-1,1)

	##########Bayesian Optimization#########
	while iteration < iterations:
		new_time =time.time()
		#### Delete Everything inside Results folder so we don't run out of space #######
		if os.path.exists(directory_R1):
			delete_folder(directory_R1)
		##########Make and solve next set###########
		if iteration != 0:
			## Running the next set of sims
			print(f"Performing iteration number {iteration}",flush=True)
			generate_geometries(unnormalize(NcandidateX,bounds),directory_O,dim,batch_size,simType,data["simSettings"]["simGeoJson"],data["simSettings"]["simConfigJson"],data["fileDirectory"]["resultFolder"],data["simSettings"]["numProcsPerSim"],data["simSettings"]["simMeshFile"])
			Q_val,C_val,NcandidateX,Participation_Data2=run_simulations(parallel_cur,batch_size,NcandidateX,simType,data["simBounds"]["tanD"],directory_O) 
			NC_Q=torch.tensor(Q_val,dtype=dtype).reshape(-1,1)
			NC_C=torch.tensor(C_val,dtype=dtype).reshape(-1,1)
					
			## Understanding how well the new points are
			value_gp=model.models[0]
			constraint_gp=model.models[1]
			with torch.no_grad():
				q_post=value_gp.posterior(NcandidateX)
				pred_Q=q_post.mean.squeeze(-1)
				std_Q=q_post.variance.sqrt().squeeze(-1)
				
				c_post=constraint_gp.posterior(NcandidateX)
				pred_C=c_post.mean.squeeze(-1)
				std_C=c_post.variance.sqrt().squeeze(-1)
			
			print(f"{'Pt':>8}{'Pred_Q':>12}{'Q_std':>12}{'Act_Q':>12}{'Perc_Q':>12}{'Pred_C':>12}{'C_std':>12}{'Act_C':>12}{'Perc_C':>12}{'dimX (um)':>12}{'dimY (um)':>12}")
			print("-"*120)
			test_print=unnormalize(NcandidateX,bounds)
			test_xlen=2*test_print[:,Pdict["PadX"]["index"]]+2*test_print[:,Pdict["Junc_len"]["index"]]+2*test_print[:,Pdict["P2GndX"]["index"]]+portlen
			test_ylen=test_print[:,Pdict["PadY"]["index"]]+2*test_print[:,Pdict["P2GndY"]["index"]]
			test_ylen=test_ylen.reshape(-1,1)
			test_xlen=test_xlen.reshape(-1,1)
			for i in range(NcandidateX.shape[0]):
				print(
					f"{i:2d}    "
					f"{pred_Q[i].item():12.4f} "
					f"{std_Q[i].item():12.4f} "
					f"{NC_Q[i].item():12.4f} "
					f"{((pred_Q[i].item()-NC_Q[i].item())/NC_Q[i].item()*100):12.4f} "
					f"{pred_C[i].item():12.4f} "
					f"{std_C[i].item():12.4f} "
					f"{NC_C[i].item():12.4f} "
					f"{((pred_C[i].item()-NC_C[i].item())/NC_C[i].item()*100):12.4f} "
					f"{test_xlen[i].item():12.4f}"
					f"{test_ylen[i].item():12.4f}"
					
				)
			## Updating the full training set
			Ntrain_x=torch.cat([Ntrain_x,NcandidateX])
			N_Q=torch.cat([N_Q,NC_Q])
			N_C=torch.cat([N_C,NC_C])
			Participation_Data = np.concatenate((Participation_Data,Participation_Data2),axis=1)
			
			
		mask=torch.ones(Ntrain_x.shape[0],dtype=torch.bool)
		for indices,coefficients,rhs in inequality_constraints:
			values = Ntrain_x[:,indices] @ coefficients
			mask &= values >= rhs-1e-8
		feasible = (torch.abs(N_C-C_target)<=C_tol).flatten() & mask.flatten()
		print(f"Total Number of feasible points: {feasible.sum()}",flush=True)	
		if feasible.any():
			Q_filGK=N_Q[feasible.flatten()]
			C_filGK=N_C[feasible.flatten()]
			X_filGK=Ntrain_x[feasible.flatten()]	
			Q_Best=Q_filGK.max()
			X_Best=unnormalize(X_filGK[torch.argmax(Q_filGK)],bounds)
			C_Best=C_filGK[torch.argmax(Q_filGK)]
			if Old_bQ==Q_Best.item():
				print(f"Best Design from FEM unchanged at Q= {Q_Best.item()} on iter= {Old_bQ_iter}",flush=True)
				print(f"Best Design from FEM Capacitance C= {C_Best.item()} ",flush=True)
				print(f"Best Design parameters {Old_bX}")
			else:
				print(f"New Best Design from FEM Q= {Q_Best.item()} with an improvement of {Q_Best.item()-Old_bQ} over result from iter= {Old_bQ_iter}",flush=True)
				print(f"Best Design from FEM Capacitance C= {C_Best.item()} ",flush=True)
				print(f"Best Design parameters {X_Best}")
				if iteration!=0 and detectMeshSpike:
					print("Checking around maxima before training GP")
					new_near=nearby_points(X_Best,percent=0.005,n_points=batch_size,lower=bounds[0,:],upper=bounds[1,:],param_range=bounds[1,:].numpy()-bounds[0,:].numpy())
					Nnew_near=normalize(new_near,bounds)
					generate_geometries(new_near,directory_O,dim,batch_size,simType,data["simSettings"]["simGeoJson"],data["simSettings"]["simConfigJson"],data["fileDirectory"]["resultFolder"],data["simSettings"]["numProcsPerSim"],data["simSettings"]["simMeshFile"])
					Q_val,C_val,_,_=run_simulations(parallel_cur,batch_size,Nnew_near,simType,data["simBounds"]["tanD"],directory_O) 
					check_NC_Q=torch.tensor(Q_val,dtype=dtype).reshape(-1,1)
					if np.abs(Q_Best.item()-check_NC_Q.mean().item())<=0.2*np.abs(Q_Best.item()):
						print("No Meshing Spike Detected",flush=True)
						Old_bQ=Q_Best.item()
						Old_bX=X_Best.numpy()
						Old_bQ_iter=iteration
					else:
						print("Meshing Spike Detected, Excluding from training",flush=True)
						check_idx=torch.where(torch.abs(N_Q.squeeze()-Q_Best)<1e-6)[0]
						mask=torch.ones(Ntrain_x.shape[0],dtype=torch.bool)
						mask[check_idx.item()]=False
						N_Q=N_Q[mask]
						N_C=N_C[mask]
						Ntrain_x=Ntrain_x[mask]
						Participation_Data=Participation_Data[:,mask]
						
		else:
			raise RuntimeError("No valid points. Stopping simulation. Either increase intial points or set more feasible targets")

		# Train GP for Q
		gp_Q=SingleTaskGP(Ntrain_x,N_Q,outcome_transform=Standardize(m=1))
		mll_Q=ExactMarginalLogLikelihood(gp_Q.likelihood,gp_Q)
		fit_gpytorch_mll(mll_Q)
		
		# Train GP for C
		gp_C=SingleTaskGP(Ntrain_x,N_C,outcome_transform=Standardize(m=1))
		mll_C=ExactMarginalLogLikelihood(gp_C.likelihood,gp_C)
		fit_gpytorch_mll(mll_C)
		
		# Make Dual Model
		model=ModelListGP(
			gp_Q,
			gp_C
		)
		
		## Save GP for this iteration
		print(f" Saving Iteration: {iteration}",flush=True)
		checkpoint = {
			"iteration":iteration,
			"model_state_dict":model.state_dict(),
			"train_x":Ntrain_x,
			"N_Q":N_Q,
			"N_C":N_C,
			"bounds":bounds,
			"C_target":C_target,
			"C_tol":C_tol,
			"batch_size":batch_size,
			"inequality_constraints":inequality_constraints,
			"max_X":max_X,
			"max_Y":max_Y,
			"Participation_Data":Participation_Data
			
		}
		torch.save(checkpoint,f"{directory_R2}/gp_iter_{iteration}.pt")
		
		## Construct Acquisition Function
		EI=qLogExpectedImprovement(
			model=model,
			best_f=Q_Best,
			objective=q_objective,
			constraints=[
				cap_constraintU,cap_constraintL
			]
		)
		## Get next points by optimizing acquisition func
		print(f"Getting next points",flush=True)
		NcandidateX,acq_value=optimize_acqf(
			EI,
			bounds=torch.stack([torch.zeros(Ntrain_x.shape[-1],dtype=dtype),
			torch.ones(Ntrain_x.shape[-1],dtype=dtype)
			]),
			q=batch_size,
			num_restarts=res_GK,
			raw_samples=raw_GK,
			inequality_constraints=inequality_constraints,
		)

		print(f"Belief for next iteration improvement: {np.exp(acq_value.item())}")
		if iteration == 0:
			print(f"Iteration {iteration} took {time.time()-first_time:.3f} s")
		else:
			print(f"Iteration {iteration} took {time.time()-new_time:.3f} s")
		
		iteration+=1

	if data["simSettings"]["meshSpikeCheck"]:
		print("Performing meshSpikeCheck by sweeping GP")
		predict_Q=make_predict_Q(model.models[0])
		predict_Qstd=make_predict_Qstd(model.models[0])	
		predict_C=make_predict_C(model.models[1])
		predict_Cstd=make_predict_Cstd(model.models[1])

		CC_U=make_CC_U2(C_target,C_tol)
		CC_L=make_CC_L2(C_target,C_tol)
		CC_arr=[
			{"type":"ineq","fun":CC_L},
			{"type":"ineq","fun":CC_U}
		]

		for indices,coeff,rhs in inequality_constraints:
			indices=indices.cpu().numpy()
			coeff=coeff.cpu().numpy()
			rhs=rhs.item() if torch.is_tensor(rhs) else rhs
			
			CC_arr.append({
				"type":"ineq",
				"fun" : lambda x, indices=indices,coeff=coeff,rhs=rhs: x[indices]@ coeff - rhs
			
			})
		sampler = qmc.Sobol(d=dim, scramble=True)
		starts = sampler.random_base2(m=15)
		GP_search_val=None
		for x0 in starts:
			res = minimize(
				objective,
				x0,
				method="SLSQP",
				bounds=[(0,1)]*dim,
				constraints=CC_arr
			)
			if not res.success:
				continue
			feasible = all(c['fun'](res.x) >= -1e-8 for c in CC_arr)
			if not feasible:
				continue
			if GP_search_val==None or res.fun < GP_search_val.fun:
				GP_search_val=res
		if not GP_search_val==None:
			X_max=GP_search_val.x
			Q_max=predict_Q(X_max)
			C_max=predict_C(X_max)
			print(f"Best Design Param: {unnormalize(torch.tensor(X_max),bounds)}")
			print(f"Quality factor: {Q_max}",flush=True)
			print(f"Capacitance: {C_max}",flush=True)
		else:
			print("meshSpikeCheck found no valid points")


if __name__ == "__main__":
    run_main("./Bayesian_Optimization/Files/BO_settings.json")