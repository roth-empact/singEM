import numpy as np
import meshio
import netCDF4
import sys
import argparse
sys.path.append("./")
from readgmshMesh import readGmsh


def gmshToExo(gmshFile, exoFile):
  """ 
  Helper function for converting a gmsh file into an exodus file for plotting in Paraview

  Input:
    gmshFile - Existing path to the gmsh (.msh) mesh file
  Output:
    exoFile  - Desired path to the exodus (.e) mesh file
  """

  nodeList, tetList, wedgeList, volStartInds, volEleTypes, triList, quadList, surfStartInds, surfEleTypes, lineList, lineStartInds, blockNames = readGmsh(gmshFile, getBlockNames=True)
  points = nodeList[:,1:] # Points for each node

  ## Group all cells into a list
  cells = []

  ## Add each volume block
  for ii in range(len(volStartInds)-1):
    if volEleTypes[ii] == "tetra":
      cells.append((f"{volEleTypes[ii]}", tetList[volStartInds[ii]:volStartInds[ii+1]]))
    elif volEleTypes[ii] == "wedge":
      cells.append((f"{volEleTypes[ii]}", wedgeList[volStartInds[ii]:volStartInds[ii+1]]))
  if len(volStartInds) > 0:
    # Add the final volume block
    if volEleTypes[-1] == "tetra":
      cells.append((f"{volEleTypes[-1]}", tetList[volStartInds[-1]:]))
    elif volEleTypes[-1] == "wedge":
      cells.append((f"{volEleTypes[-1]}", wedgeList[volStartInds[-1]:]))

  ## Add each surface block
  for ii in range(len(surfStartInds)-1):
    if surfEleTypes[ii] == "tri":
      cells.append((f"triangle", triList[surfStartInds[ii]:surfStartInds[ii+1]]))
    elif surfEleTypes[ii] == "quad":
      cells.append((f"{surfEleTypes[ii]}", quadList[surfStartInds[ii]:surfStartInds[ii+1]]))
  if len(surfStartInds) > 0:
    # Add the final surfume block
    if surfEleTypes[-1] == "tri":
      cells.append((f"triangle", triList[surfStartInds[-1]:]))
    elif surfEleTypes[-1] == "quad":
      cells.append((f"{surfEleTypes[-1]}", quadList[surfStartInds[-1]:]))

  ## Add each line block
  for ii in range(len(lineStartInds)-1):
    cells.append((f"line", lineList[lineStartInds[ii]:lineStartInds[ii+1]]))
  if len(lineStartInds) > 0:
    # Add the final line block
    cells.append((f"line", lineList[lineStartInds[-1]:]))

  mesh_new = meshio.Mesh(
    points,
    cells=cells
  )

  # # Delete fields that don't cooperate with vtk
  # mesh.point_data = {}
  # mesh.cell_sets = {}
  # cellFields = ['gmsh:physical', 'gmsh:geometrical']
  # for field in cellFields:
  #   if field in mesh.cell_data:
  #     for ii in range(len(mesh.cell_data[field])):
  #       # Cast to a generic int to prevent 'vtktypeint32' formatting strings
  #       mesh.cell_data[field][ii] = mesh.cell_data[field][ii].astype(int)

  meshio.write(exoFile, mesh_new)

  ## Reopen the exodus file with netCDF4 and add in block names
  with netCDF4.Dataset(exoFile, "r+") as ds:
    if "eb_names" not in ds.variables:
      if "len_name" not in ds.dimensions:
        # 33 is the standard historical character limit for Exodus names
        ds.createDimension("len_name", 33) 
      ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))
    
    # Write strings converted to bytes to the eb_names array
    for ii, name in enumerate(blockNames):
      # netCDF character arrays need padding to match the max string length dimension
      name_bytes = name.encode('utf-8')
      name_enc = [_i.encode() if hasattr(_i, "encode") else _i for _i in name] # Encoded version of block name
      ds.variables["eb_names"][ii, :len(name_bytes)] = name_enc 


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                    prog='main.py',
                    description='Main wrapper for finite element simulations.')
  parser.add_argument('-g', '--gmshFile',
                      help="Path to the gmsh file to read from. Should end with .msh extension",
                      default='./Meshes/Qubit/qubit_10umTrench.msh')
  parser.add_argument('-e', '--exoFile', 
                      help="Path to the exodus file to write to. Should end with .e extension.",
                      default='./Meshes/Qubit/qubit_10umTrench.e')
  args = parser.parse_args()

  gmshToExo(args.gmshFile, args.exoFile)