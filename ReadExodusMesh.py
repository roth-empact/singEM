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

import netCDF4 as ncd
import numpy as np
# import meshio as msh

def readEx(fileName):
  '''
  Reads Exoodus (.e) file and returns information about the mesh. Requires 
  use of netCDF4 package easily downloadable with instructions from the netCDF4 feedstock
  spinoff version of readAbq method set to read Abaqus files but using Exodus files for more efficient
  file transfer/translation
  
  nodes     - node numbers and coordinates of each node
  elements  - element number and node indices for each tetrahedral element, followed by permittivity and permeability
  note: all information is valid, but the nodes list will be split up inot a series of connectivity arrays
  which does hold correct information relative to the given node map. 
  for more information, calling ncdump filename will give all outputs of raw entries and values.
  ncdump -h gives general header information of variables, etc.
  To whoever might be picking up the code: ncdump will give you more information than anything you can find online. However,
  for some general knowledge you can reference: https://johnfoster.pge.utexas.edu/blog/posts/extracting-exodus-information-with-netcdf-python/
  or for some more standardized knowledge: https://sandialabs.github.io/seacas-docs/exodusII-new.pdf.
  IMPORTANT: make sure to use Anaconda prompt through anaconda to conda forge download the netcdf package as shown in imports. You can 
  search up netcdf python feedstock the see how to download the package

  Authors: Jinhoo Yoon (Feel free to add your name here if you're contributing to the code)
  '''

  nc = ncd.Dataset(fileName)
  blockCnt = len(nc.variables["eb_names"]) # blockCnt is the number of element blocks
  x = nc.variables['coordx'] # gets coordinates in x,y, and z
  y = nc.variables['coordy']
  z = nc.variables['coordz']
  tets = np.zeros((0, 4), dtype=np.int32)
  wedges = np.zeros((0, 6), dtype=np.int32)
  tris = np.zeros((0, 3), dtype=np.int32)
  quads = np.zeros((0, 4), dtype=np.int32)
  lines = np.zeros((0, 2), dtype=np.int32)
  volStartInds = []
  volEleTypes = []
  surfStartInds = []
  surfEleTypes = []
  lineStartInds = []
  for i in range(blockCnt):
    c_name = f'connect{i+1}'
    c_temp = np.asarray(nc.variables[c_name]) # actual connectivity array for given element
    elem_type = nc.variables[c_name].elem_type
    if (elem_type == 'TETRA'):
      volEleTypes.append('Tet')
      volStartInds.append(len(tets))
      tets = np.append(tets, c_temp, axis=0)
    elif elem_type == 'WEDGE':
      volEleTypes.append('Wedge')
      volStartInds.append(len(wedges))
      wedges = np.append(wedges, c_temp, axis=0)
    elif (elem_type == 'TRI3' or elem_type == 'TRI'): 
      surfStartInds.append(len(tris))
      tris = np.append(tris, c_temp, axis=0)
    elif elem_type == 'BAR2' or elem_type == 'B21':
      lineStartInds.append(len(lines))
      lines = np.append(lines, c_temp, axis=0)


  nodes = np.array([np.arange(len(x)), x, y, z]).T

  nc.close() # closes the netcdf file (not exactly but close enough for implementation purposes)
  return nodes, tets, wedges, volStartInds, volEleTypes, tris, quads, surfStartInds, surfEleTypes, lines, lineStartInds

def writeExResults(fileName, resData, varName):
  """
  Writes a vector of simulation results into an exodus file so it can be input into other tools (Paraview, Cubit)

  Inputs:
    fileName - The path to the Exodus file that the results will be written into
    resData  - Data associated with each element (currently only tetrahedrons are supported)
    varName  - Name of the variable to write into the file
  """
  
  varNames = [varName + "_x", varName + "_y", varName + "_z"]

  ## Load in the exodus file containing the mesh
  nc = ncd.Dataset(fileName, "a")
  
  ## Set up times so time-dependent variables can be instantiated
  time_var = nc.variables['time_whole']
  time_steps = (int)(resData.shape[1] / 3)
  time_var[:] = np.arange(time_steps)

  for t in range(time_steps):
    for i in range(3):   
      varName_enc = [_i.encode() if hasattr(_i, "encode") else _i for _i in varNames[i]] # Encoded version of variable name
  
      ## Create an element-based time-dependent variable (if not already present)
      newVar = True # Marks whether the variable is new
      if "name_elem_var" not in nc.variables:
        varCnt = 0
      else: # If one is already present, move it so we can make a new one with the correct size
        varCnt = nc.dimensions["num_elem_var"].size
        for ii in range(varCnt): # Make sure the variable isn't already defined
          if np.all(nc.variables["name_elem_var"][ii, :len(varNames[i])] == varName_enc):
            varCnt = ii
            newVar = False
            break
        if newVar:
          # print(f"Creating variable {varNames[i]} at index {varCnt}")
          nc.renameVariable('name_elem_var', f'old_names_{varCnt}')
          nc.renameDimension('num_elem_var', f'old_num_{varCnt}')
      if newVar:
        # print(f"BOOM Creating variable {varNames[i]} at index {varCnt}")
        nc.createDimension('num_elem_var', varCnt+1)
        nc.createVariable('name_elem_var', "S1", ('num_elem_var', 'len_name'))
        if varCnt != 0: # Copy all other variables over if this isn't the first one
          nc.variables["name_elem_var"][:varCnt] = nc.variables[f"old_names_{varCnt}"][:varCnt]
        nc.variables["name_elem_var"][varCnt] = b""
        nc.variables["name_elem_var"][varCnt, :len(varNames[i])] = varName_enc
  
      ## Set values for the variable across each block (only nonzero for volume blocks)
      blkStart = 0
      for blkNum in range(nc.dimensions['num_el_blk'].size):
        ## Create an element-based variable for the current block (if not already present)
        if f"vals_elem_var{varCnt+1}eb{blkNum+1}" in nc.variables:
          res = nc.variables[f"vals_elem_var{varCnt+1}eb{blkNum+1}"]
        else:
          res = nc.createVariable(f'vals_elem_var{varCnt+1}eb{blkNum+1}', 'f8', ('time_step',f'num_el_in_blk{blkNum+1}'))

        ## Check if the block contains volume elements before adding in values (tet = 4 nodes, wedge = 6 nodes)
        if nc.dimensions[f"num_nod_per_el{blkNum+1}"].size == 4 or nc.dimensions[f"num_nod_per_el{blkNum+1}"].size == 6:
          res[t] = resData[blkStart:blkStart+nc.dimensions[f'num_el_in_blk{blkNum+1}'].size, 1 + t * 3 + i]
          blkStart += nc.dimensions[f'num_el_in_blk{blkNum+1}'].size
        else:
          res[t] = np.zeros((nc.dimensions[f'num_el_in_blk{blkNum+1}'].size,), dtype=np.float64)

  nc.close()

if __name__ == "__main__":
  meshfile = "./Meshes/Microstrips/microstrip.e"
  # meshfile = "./Meshes/params_out.e"
  
  #writeExResults(meshfile)







