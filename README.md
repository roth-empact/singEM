# singEM: A Singularity-Enriched Finite Element Solver for Electromagnetics

This document is intended to provide a basic overview of the features of the singEM package. This pacakge is primarily intend for performing electromagnetic (EM) simulations of microwave circuitry using the finite element method (FEM), with a particular emphasis on superconducting qubit geometries. To extract critical quantities such as participation ratios for these devices accurately and efficiently, this package uses specialized singular basis functions to model singular fields near metal corners[^1]. When combined with optimization techniques such as Bayesian optimization (BO), this technique can be used to rapidly design qubits with high coherence times at relatively low computational cost.

This document will use examples to demonstrate how simulations can be configured in this package, and how the outputs can be post-processed to obtain useful figures of merit and plots of the underlying fields.
# Environment Setup

### Windows

Simulations using this package can be run locally on Windows systems, but they will not be able to access efficient parallel solvers provided by the [PETSc package](https://petsc.org/release/). I
To set up an anaconda environment to use the package on Windows, you can run the following command in a terminal with [Anaconda](https://www.anaconda.com/download/success?reg=skipped) activated:

``` Bash
conda env create -f ./Bash_Scripts/singEM_windows.yml
```

Once the environment is set up, you can activate it and run simulations. See the "Rectangular Cavity" example in the Examples section below for instructions on how to run a basic simulation using this environment.

The examples will also show how [Paraview](https://www.paraview.org/) can be used to visualize simulations results. If paraview is installed, it can be connected to this package by changing the Paraview path in [pvController.py](./PV_visualization/pvController.py). The examples will expand further upon this functionality.
## Linux (Cluster)


Most applications of this package will require using a distributed computing cluster. This section will assume that users have some basic background with cluster usage, and will assume that the cluster is using SLURM for job scheduling. Similarly to the above, the user should run the following command within a job:

``` Bash
conda env create -f ./Bash_Scripts/singEM_linux.yml
```

An example job file is included in [envSetup_linux.sh](./Bash_Scripts/envSetup_linux.sh) (note: you may need to run `dos2unix ./Bash_Scripts/envSetup_linux.sh` prior to submitting this job file). Once the environment is set up, successive jobs can simply activate the environment prior to running main.py from this package. More details on simulation configuration will be provided in the examples below.
 

#  Examples

In the following examples, we walk through the steps to set up simulations of various electromagnetic geometries and analyze their outputs.

## Rectangular Cavity (Basic Eigenmode Simulation)

We first look at an example of how to extract the eigenmodes of a simple rectangular cavity using SciPy. This example can be performed locally on a Windows machine or on a cluster, but we will assume it is being done locally. Subsequent examples will use PETSc for more efficient solutions.

In general, the simulation is configured by running main with the flag "-f" pointing to a simulation configuration file.The primary configuration file used to specify the settings used to run this example is [cavity_Eig.json](./Meshes/Cavities/cavity). The contents of the file are included below, with comments indicating what each setting does:

```Python
{
  "probP": { # Problem parameters
    "type": "Eig", # The type of problem to solve ("Eig" for eigenmode, "FD" for driven frequency domain)
    "numEig": 5, # Number of eigenmodes to evaluate
    "geomFile": "./Meshes/Cavities/recCavity.json", # Path to a "geometry configuration file" containing 
                                                    # material properties and boundary conditions
    "meshFile": "./Meshes/Cavities/recCavity.msh", # Path to the mesh file exported from a meshing tool 
                                                   # (gmsh or Cubit)
    "bOrder": 1 # Order of the basis functions (can be 1 or 2)
  },
  "solveP": { # Solver parameters
    "pkg": "SciPy", # Package used to perform the solve (either "SciPy" or "SLEPc" for eigenvalues)
    "shift": 20e9, # Value (in Hz) to look for eigenvalues near
    "tol": 1e-06 # Desired tolerance for the eigenvalues
  },
  "outP": { # Output parameters
    "resFile": "./Results/recCavity_p=1.npz", # File to save the results in (must be .npz)
    "saveFields": true, # Flag indicating whether to plot the eigenvectors in all elements 
    "fieldType": "E", # Which field to plot if "saveFields" == True
    "fFile": "fields.npy" # File containing field information for eigenvectors
  },
  "numProcs": 1 # Number of parallel processes to run the simulation with (1 = serial)
}
```
Note that a complete list of settings can be found in the docstring of `writeSimJSON()` in [JSON_util.py](./JSON_util.py). 

In order to specify the properties of the volume of the cavity and its outer boundary, we include the "geomFile" above. The file [recCavity.json](./Meshes/Cavities/recCavity.json) is also included below to demonstrate how these files can be configured. The mesh file [recCavity.msh](./Meshes/Cavities/recCavity.msh) contains two "blocks": one containing all tetrahedrons in the volume of the cavity, and one containing all triangles on the outer boundary of the cavity. As shown below, the geometry configuration file gives the properties of each of these blocks. 

```Python
{
  "scale": 1, # Scaling applied to the dimensions used in the mesh 
              # If the mesh is defined in mm, this should be set to 1e-3
  "volP": { # Parameters of volume blocks
    "num": 1, # Number of volume blocks
    "vol0": { # Properties of volume block 0
      "epr": 1.0, # Relative permittivity
      "mur": 1.0, # Relative permeability
      "sigma": 0.0, # Conductivity
      "tanD": 0.0 # Loss tangent
    }
  },
  "surfP": { # Parameters of surface blocks
    "num": 1, # Number of surface blocks
    "surf0": {
      "type": "Outer" # Type of boundary condition (outer = outer boundary, which is mapped to PEC)
    }
  }
}
```
As for the simulation files, more detailed documentation on the types of blocks and their available settings can be found in the docstring of `writeGeometryJSON()` in [JSON_util.py](./JSON_util.py). 

For the volume block, we specify material properties such as permittivity and permeability. For the surface, in this case we just mark this block as being the outer boundary. In this version of the package, the outer boundary is always treated as a perfect electric conductor (PEC), but later versions may add support for additional boundary conditions.

Once the settings are configured, the simulation can be ran using the following command

```Bash
python main.py -f "./Meshes/Cavities/cavity_Eig.json"
```

The results will print out in the terminal, and should look similar to the following:

```
Time spent forming element lists: 1.004 s
Number of tetrahdrons: 990
Number of wedges: 0
Number of nodes: 48
Number of edges: 771
Number of triangular faces: 1712
Number of quadrilateral faces: 0
Longest triangular edge: 2.592e+00 mm
Shortest triangular edge: 1.124e+00 mm
Average triangular edge length: 1.807e+00 mm
Triangular edge length standard deviation: 2.979e-01 mm
Time spent building basis maps: 0.005 s. Element order: 1. Singular element order: 1 Total number of unknowns: 771
Beginning matrix assembly
Matrix assembly done, time elapsed = 10.110 s
Finding eigenvalues
5 eigenvalues found in 0.057 s.
Eigenvalue 0: 24.50740+0.00000j GHz
Eigenvalue 1: 32.81014+0.00000j GHz
Eigenvalue 2: 35.03249+0.00000j GHz
Eigenvalue 3: 35.13929+0.00000j GHz
Eigenvalue 4: 37.82026+0.00000j GHz
Total run time: 11.902 s
```

After providing some diagnostic and timing information, the requested eigenvalues are printed. If the code is configured to connect to Paraview (see Windows Environment Setup above)
and `"saveFields" == true` in [cavity_Eig.json](./Meshes/Cavities/cavity_Eig.json), Paraview will open and allow the user to visualize the fields throughout the cavity. Note that this plotting method only allows a single value to be provided for each mesh element, so the resolution will be relatively poor for coarse meshes like this one. More sophisticated plotting approaches will be discussed in later examples to overcome this issue.

When Paraview initially opens, the center view may be blank. The geometry can be plotted by clicking the closed eye icon in the "Pipeline Browser" (which may need to be added via the View tab). To plot the field distribution throughout the geometry, the "coloring" drop-down in the "Properties" menu should be set to "E_" (which can be tuned to plot the magnitude or any vector component). If there are surface blocks, these need to be unchecked in the "Element Blocks" section of the "Properties" menu to ensure only data associated with the tetrahedrons is included. By changing the display representation in that menu to "Volume," we can generate images such as the one below:

![Plot of an eigenmode of the rectangular cavity](./Results/cavity_Eig.png)

For eigenmode problems, field plots for the other modes can also be loaded into paraview. These can be accessed by opening the "Time Manager" window from the View menu and changing the timestep. The view can also be modified further by applying filters such as cutting planes and thresholding functions. More details about these settings can be found in the [Paraview User's Guide](https://docs.paraview.org/en/latest/UsersGuide/index.html).

## Ridged Waveguide Cavity (Singular Eigenmode Simulation)

Next, we consider another eigenmode problem in the form of a ridged waveguide cavity. As shown in the plot below, the dominant eigenmode of this cavity is singular with respect to the edges of the ridge.

![Plot of an eigenmode of the rectangular cavity](./Results/ridgedCavity_Eig.png)

In order to resolve this mode accurately with a coarse mesh, we include singular basis functions in the tetrahedrons contacting this metal edge. In the simulation file [ridgedCavity_Eig.json](./Meshes/Cavities/ridgedCavity_Eig.json), we add another setting to specify the singular basis function order:

```Python
{
  "probP": {
    "type": "Eig",
    "numEig": 5,
    "geomFile": "./Meshes/Cavities/ridgedCavity_sing.json",
    "meshFile": "./Meshes/Cavities/ridgedCavity.e",
    "bOrder": 1,
    "sOrder": 1 # Order of singular basis functions
  },
  "solveP": {
    "pkg": "SciPy",
    "shift": 20e9,
    "tol": 1e-06
  },
  "outP": {
    "resFile": "./Results/ridgedCavity_p=1,s=1.npz",
    "saveFields": true,
    "fieldType": "E",
    "fFile": "fields.npy"
  },
  "numProcs": 1
}
```

These functions are only included in tetrahedrons contacting singular edges in the geometry. These edges are marked by blocks in the geometry, as can be seen in the geometry file [ridgedCavity_sing.json](./Meshes/Cavities/ridgedCavity_sing.json):

```Python
{
  "scale": 1,
  "volP": {
    "num": 1,
    "vol0": {
      "epr": 1.0,
      "mur": 1.0,
      "sigma": 0.0,
      "tanD": 0.0
    }
  },
  "surfP": {
    "num": 1,
    "surf0": {
      "type": "Outer"
    }
  },
  "lineP": { # Parameters of line blocks
    "num": 1, # Number of line blocks
    "line0": {
      "type": "singLine", # Marks that singular basis functions should be added
                          # to tets touching nodes or edges along this line
      "coeff": 0.667 # Coefficient of the singularity
    }
  }
}

```

In the current version of the code, integrals of the singular basis functions are pre-calculated for a unit tetrahedron and rescaled to the dimensions of each tetrahedron in the mesh[^1]. This step has been performed in advance for coefficients of 1/2 and 2/3 to match the expected behavior near metal wedges at 0 and 90 degree angles respectively [^2], and the resulting integrals are stored in [BasisMats](./BasisMats/). If alternative coefficients are needed, additional files can be generated using [singBasisMats_tets_numeric.py](singBasisMats_tets_numeric.py).

Once again, we can run the simulation as configured above using

```Bash
python main.py -f "./Meshes/Cavities/cavity_Eig.json"
```

For eigenvalue 0, the singular basis functions give an eigenvalue of `31.44299 GHz`, giving a percent error of `1.09%` relative to the value of `31.76373 GHz` obtained using 2nd order functions on a dense mesh. If we remove the singular basis functions by changing "geomFile" in [ridgedCavity_Eig.json](./Meshes/Cavities/ridgedCavity_Eig.json), the resulting eigenvalue is `30.76217 GHz`, meaning the percent error is increased to `3.15%`. A more detailed comparison of the performance can be found in our publication on the singular functions [^1].

## Microstrip (Driven Simulation)

Here, we demonstrate how to perform a driven simulation using a microstrip transmission line as an example. In this case, we set up the problem on a computing cluster and use PETSc to evaluate the linear system produced by the FEM code. The configuration file for this simulation is [microstrip_FD.json](./Meshes/Microstrip/microstrip_FD.json), and the settings are included below:

```Python
{
  "probP": {
    "type": "FD", # Problem type is frequency domain (driven)
    "minFreq": 1e9, # Minimum frequency to solve at is 1 GHz
    "maxFreq": 10e9, # Maximum frequency to solve at is 10 GHz
    "numSteps": 1001, # Number of frequency steps between the minimum and maximum
    "stepType": "linear", # Type of spacing between steps (either "linear" or "log")
    "geomFile": "./Meshes/Microstrips/microstrip.json",
    "meshFile": "./Meshes/Microstrips/microstrip.msh", 
    "outBC": "PEC",
    "bOrder": 2, # Using 2nd order basis functions for higher accuracy   
    "portMatrix": false # Flag indicating whether to simulate driving from all ports (true)
                        # or from only a single port (false). If false, uses settings from the geometry file
  },
     
  "solveP": {
    "pkg": "PETSc" # Using a solver from PETSc rather than SciPy
  },  
  "outP": {
    "saveV": true,
    "saveS": true,
    "resFile": "./Results/microstrip.npz",
    "saveFields": false
  },
  "numProcs": 32 # Solve using 32 parallel processes (job must request 32 cores)
}
```

The mesh file [microstrip.msh]("./Meshes/Microstrips/microstrip.msh") contains several blocks, including two blocks for the volume, and five surface blocks. The surface blocks include one for the outer boundary, two PEC blocks for the signal trace and ground plane, and two blocks for the ports used to drive and terminate the geometry. The file is included with comments below to demonstrate how these blocks are specified:

```Python
{
  "scale": 1,
  "volP": {
    "num": 2, # Number of volume blocks
    "vol0": { # Parameters of the substrate block
      "epr": 7.0, # Relative permittivity of the substrate
      "mur": 1.0,
      "sigma": 0.0,
      "tanD": 0.0
    },
    "vol1": { # Parameters of the free-space block
      "epr": 1.0, # Relative permittivity of free-space
      "mur": 1.0,
      "sigma": 0.0,
      "tanD": 0.0
    }
  },
  "surfP": {
    "num": 5, # Number of surface blocks
    "surf0": {
      "type": "Outer" # Outer boundary
    }, 
    "surf1": {
      "type": "PEC" # Signal trace
    }, 
    "surf2": {
      "type": "PEC" # Ground plane
    }, 
    "surf3": {
      "type": "Port", # Input port
      "Z": 50, # Port impedance (only used if portType != "open") 
      "portType": "Open", # Marks whetehr her the port is open-circuited ("Open") or resistive ("R")
                          # Inductive ("L") or capacitive ("C") ports are also available options.
                          # In these cases, "Z" gives the inductance or capacitance itself, and frequency 
                          # scaling is applied internally
      "width": 0.001, # Width of the port as defined in the mesh file
      "direction": [ # Unit vector describing the direction of the port
        0.0, 
        1.0,
        0.0
      ],
      "normal": [ # Unit vector describing the direction normal to the port
        0.0, 
        0.0,
        1.0
      ],
      "Vsmag": 1.0 # Magnitude of the voltage being sourced to this port. 
                   # If portType == "open", this value is interpreted as the magnitude of the current source.      
    },
    "surf4": { 
      "type": "Port", # Output port
      "Z": 50,
      "portType": "Open",
      "width": 0.001,
      "direction": [
        0.0,
        1.0,
        0.0
      ],
      "normal": [
        0.0,
        0.0,
        1.0
      ],
      "Vsmag": 0.0
    }
  }
}
```

[^1]: [Elkin, S. T., Khan, G., Forati, E., Langley, B. W., Timucin, D., Molavi, R., & Roth, T. E. (2026). Singular Vector Finite Element Basis Functions for Tetrahedra in Complex Electromagnetic Geometries. arXiv preprint arXiv:2606.18140.](https://arxiv.org/abs/2606.18140)

[^2]: Jackson, J. D. (1998). Classical Electrodynamics. John Wiley & Sons.

#