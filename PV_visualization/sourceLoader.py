import os
import numpy
from paraview.simple import *

# Function to read an external Python script
def read_script(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def load_script(mesh_file, data_file, slice = False, numT = 1, vector = False):

    # since paraview doesn't like importing other python modules, we link the other scripts into
    # a single string that the paraview interpreter can execute
    #linked_script = read_script('./PV_visualization/readAbaqusMesh.py')
    linked_script = read_script('./ReadAbaqusMesh.py')
    linked_script += "\n" + read_script('./PV_visualization/generateSolidMesh.py')
    linked_script = linked_script.replace('MESH_FILE', mesh_file)
    linked_script = linked_script.replace('DATA_FILE', data_file)
    if vector: # For handling the vector case correctly in generateSolidMesh()
        linked_script = linked_script.replace('VECTOR', 'TRUE') 
    else:
        linked_script = linked_script.replace('VECTOR', 'FALSE') 
    # Create a copy for wedges, and specify element type in each
    wedge_script = linked_script
    linked_script = linked_script.replace('ELE_TYPE', 'TET')
    wedge_script = wedge_script.replace('ELE_TYPE', 'WEDGE')

    info_script = read_script('./PV_visualization/requestInfo.py')
    info_script = info_script.replace('NUM_TIMESTEPS', f"{numT}")

    # Set up programmable source
    prog_source = ProgrammableSource()
    prog_source.OutputDataSetType = 'vtkUnstructuredGrid'
    prog_source.Script = linked_script
    prog_source.ScriptRequestInformation = info_script
    prog_source.UpdatePipeline()

    # Set up another source for the wedges
    wedge_source = ProgrammableSource()
    wedge_source.OutputDataSetType = 'vtkUnstructuredGrid'
    wedge_source.Script = wedge_script
    wedge_source.ScriptRequestInformation = info_script
    wedge_source.UpdatePipeline()

    # set up display
    renderView1 = GetActiveViewOrCreate('RenderView')
    programmableSource1Display = GetDisplayProperties(prog_source, view=renderView1)
    programmableSource2Display = GetDisplayProperties(wedge_source, view=renderView1)

    # render basic mesh with cell values
    cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=prog_source)
    cellDatatoPointData2 = CellDatatoPointData(registrationName='CellDatatoPointData2', Input=wedge_source)
    display = Show(cellDatatoPointData1, renderView1, 'UnstructuredGridRepresentation')
    display = Show(cellDatatoPointData2, renderView1, 'UnstructuredGridRepresentation')
    Hide(prog_source, GetActiveView())
    display.Representation = 'Volume'
    display.RescaleTransferFunctionToDataRange(False, True)
    if (vector):
        ColorBy(display, ('POINTS', 'fields', 'magnitude'))
    else:
        ColorBy(display, ('POINTS', 'fields'))

    renderView1.Update()
    Render()

    if slice:
        Hide(cellDatatoPointData1, GetActiveView())

        bounds = prog_source.GetDataInformation().GetBounds()

        # create first slice
        slice1 = Slice(registrationName='Slice1', Input=cellDatatoPointData1)
        slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')
        slice1Display.Representation = 'Surface'
        slice1Display.SetScalarBarVisibility(renderView1, True)

        slice1.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        slice1.SliceType.Normal = [1.0, 0.0, 0.0]

        # create second slice
        slice2 = Slice(registrationName='Slice2', Input=cellDatatoPointData1)
        slice2Display = Show(slice2, renderView1, 'GeometryRepresentation')
        slice2Display.Representation = 'Surface'
        slice2Display.SetScalarBarVisibility(renderView1, True)

        slice2.SliceType.Origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        slice2.SliceType.Normal = [0.0, 0.0, 1.0]

    if vector:

        glyph1 = Glyph(registrationName='Glyph1', Input=GetActiveSource(),
            GlyphType='Arrow')
        glyph1.ScaleArray = ['POINTS', 'fields']
        glyph1.OrientationArray = ['POINTS', 'fields']
        glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
        glyph1Display.Representation = 'Surface'
        glyph1Display.SetScalarBarVisibility(renderView1, True)
        renderView1.Update()

# read args

with open('args', 'r') as file:
    args = file.read().split(',')
os.remove('args')

mesh_file = args[0]
data_file = args[1]
numT = int(args[2])
vector = (args[3] == "True")

# with open(data_file, 'rb') as f:
#     fields = numpy.load(f)
#     vector = (fields.shape[1] == 3)
if(os.path.splitext(mesh_file)[1] == '.inp'):
    load_script(mesh_file, data_file, False, numT, vector)
    Interact()
else:
    ExodusIIReader(FileName = mesh_file)
    Interact()