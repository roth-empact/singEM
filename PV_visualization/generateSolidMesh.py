from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support as nps
import numpy as np

# For timestepping, see https://docs.paraview.org/en/latest/ReferenceManual/pythonProgrammableFilter.html#reading-a-csv-file-series for details
def GetUpdateTimestep(algorithm): 
    """Returns the requested time value, or None if not present"""
    executive = algorithm.GetExecutive()
    outInfo = executive.GetOutputInformation(0)
    return outInfo.Get(executive.UPDATE_TIME_STEP()) \
              if outInfo.Has(executive.UPDATE_TIME_STEP()) else None

req_time = GetUpdateTimestep(self)
# load in mesh and data
with open('DATA_FILE', 'rb') as f:
    fields = numpy.load(f)

nodeList, tetList, wedgeList, volStartInds, volEleTypes, triList, surfStartInds, lineList, lineStartInds, _, _ = readAbq("MESH_FILE")
tetList = np.array(tetList)
if len(tetList) != len(fields): # More than just tetrahedrons are present, so account for wedges
    wedgeList = np.array(wedgeList)


# create points
points = vtk.vtkPoints()

# Create a cell array to store the tets, and another for wedges
cellArray = vtk.vtkCellArray()
cellArray.Allocate(len(tetList), 1000)

# Placeholder point to ensure indexing aligns with node id
points.InsertNextPoint(0, 0, 0)

# add nodes to vtk mesh
for node in nodeList:
    points.InsertNextPoint(node[1], node[2], node[3])

# add tets or wedgeList to vtk mesh
if "ELE_TYPE" == "TET":
    for elm in tetList:
        tetra = vtk.vtkTetra()
        for i in range(4):
            tetra.GetPointIds().SetId(i, elm[i])
        cellArray.InsertNextCell(tetra)
    
    # set up mesh
    unstructuredGrid = vtk.vtkUnstructuredGrid()
    unstructuredGrid.SetPoints(points)
    unstructuredGrid.SetCells(vtk.vtkTetra().GetCellType(), cellArray)
    fieldRange = np.arange(len(tetList), dtype=np.int32)
elif "ELE_TYPE" == "WEDGE":
    for elm in wedgeList:
        wedge = vtk.vtkWedge()
        for i in range(6):
            wedge.GetPointIds().SetId(i, elm[i])
        cellArray.InsertNextCell(wedge)

    # set up mesh
    unstructuredGrid = vtk.vtkUnstructuredGrid()
    unstructuredGrid.SetPoints(points)
    unstructuredGrid.SetCells(vtk.vtkWedge().GetCellType(), cellArray)
    fieldRange = len(tetList)+np.arange(len(wedgeList), dtype=np.int32)

# assuming fields array is in order, no gaps
num_components = fields.shape[1] - 1

# convert numpy array and assign the vtk data
# if "VECTOR" == "FALSE": 
#     if req_time != None:
#         celldata = unstructuredGrid.GetCellData()
#         vtk_data_array = nps.numpy_to_vtk(fields[fieldRange,int(req_time)+1].ravel(), deep=True)
#         vtk_data_array.SetNumberOfComponents(1)
#         vtk_data_array.SetName(f'fields')
#         celldata.AddArray(vtk_data_array)
# else:
#     celldata = unstructuredGrid.GetCellData()
#     for ii in range(num_components):
#         vtk_data_array = nps.numpy_to_vtk(fields[fieldRange,ii+1].ravel(), deep=True)
#         vtk_data_array.SetNumberOfComponents(1)
#         vtk_data_array.SetName(f'fields_{ii}')
#         celldata.AddArray(vtk_data_array)
        
#     vtk_data_array = nps.numpy_to_vtk(fields[fieldRange,1:(num_components + 1)].ravel(), deep=True)
#     vtk_data_array.SetNumberOfComponents(num_components)
#     vtk_data_array.SetName('fields')
#     celldata = unstructuredGrid.GetCellData()
#     celldata.AddArray(vtk_data_array)

if req_time != None:
    if "VECTOR" == "FALSE": 
        celldata = unstructuredGrid.GetCellData()
        vtk_data_array = nps.numpy_to_vtk(fields[fieldRange,int(req_time)+1].ravel(), deep=True)
        vtk_data_array.SetNumberOfComponents(1)
        vtk_data_array.SetName(f'fields')
        celldata.AddArray(vtk_data_array)
    else:
        celldata = unstructuredGrid.GetCellData()
        for ii in range(3):
            vtk_data_array = nps.numpy_to_vtk(fields[fieldRange,3*int(req_time)+ii+1].ravel(), deep=True)
            vtk_data_array.SetNumberOfComponents(1)
            vtk_data_array.SetName(f'fields_{ii}')
            celldata.AddArray(vtk_data_array)
            
        curTimeRange = 3*int(req_time) + np.arange(1,4)
        vtk_data_array = nps.numpy_to_vtk(fields[fieldRange][:,curTimeRange], deep=True)
        vtk_data_array.SetNumberOfComponents(3)
        vtk_data_array.SetName('fields')
        # celldata = unstructuredGrid.GetCellData()
        celldata.AddArray(vtk_data_array)

# output of the programmable source is the created unstructured grid
self.GetOutput().ShallowCopy(unstructuredGrid)
