import meshio
import numpy as np

'''

This function reads an gmsh (.msh) file and returns connectivity information about the mesh
    Inputs:
        fName         - filepath for .msh file
        getBlockNames - Flag indicating whether to return block names (for plotting purposes)

    Outputs:
        nodes         - node numbers and coordinates of each node
        tets          - element number and node indices for each tetrahedral element
        wedges        - element numbers and node indices for "wedge" (triangular prism) elements
        volStartInds  - indices marking the start of each new block (for marking material parameters)
        volEleTypes   - element type associated with each start index in 'volStartInds'
        tris          - node indices for triangles on each surface
        quads         - node indices for quadrilaterals on each surfaces
        surfStartInds - indices marking the start of each new block (for marking surface types) 
        surfEleTypes  - element type associated with each start index in 'surfStartInds'
        lines         - node indices forming each line element
        lineStartInds - indices marking the start of each new block of line elements
        blockNames    - Names of the blocks as defined in gmsh (only returned if getBlockNames == True)
'''

def readGmsh(filename, getBlockNames=False):
    mesh = meshio.read(filename)
    points = mesh.points
    nodes = []
    wedges = np.zeros((0,6), dtype=np.int32)
    tets = np.zeros((0,4), dtype=np.int32)
    tris = np.zeros((0,3), dtype=np.int32)
    quads = np.zeros((0,4), dtype=np.int32)
    lines = np.zeros((0,2), dtype=np.int32)
    volStartInds = []
    volEleTypes = []
    surfStartInds = []
    surfEleTypes = []
    lineStartInds = []
    blockNames = []

    # Storing the nodes and their indices
    nodes = np.hstack([np.arange(len(points))[:,None], points])

    # Sorting the group_name alphabetically for the JSON
    #sort_dict = dict(sorted(mesh.cell_sets.items()))
    # sort_dict = dict(mesh.cell_sets.items())
    # sort_dict = dict(mesh.cell_sets.items())
    sort_dict = dict(
    sorted(
        (
            (name, cells)
            for name, cells in mesh.cell_sets.items()
            if name in mesh.field_data
        ),
        key=lambda item: mesh.field_data[item[0]][0]
    )
    )

    # Variable to keep track of the new element "label"
    iter = 0

    # Iterating through each group. Ex: EB2, mpr1epr1 etc etc.
    for group_name, cells_list in sort_dict.items():
        # Unknown group_name. Don't think it is of any importance
        if group_name == 'gmsh:bounding_entities':
            continue
        blockNames.append(group_name)
        
        iter = 1
        # Known group_names. Iterating through the shape (cell type) each group name is made of
        for cells_index in range(len(cells_list)):
            if (len(cells_list[cells_index]) == 0):
                continue
            
            # Checking the cell types (Tet, tri, wedge, or line) and adding them to their array.
            # Also updating the indices for the start of each new cell.

            # The cells_list maps to mesh.cells. cells_list only contains cells (as list) which belongs to the 
            # which belongs to the  the specific group name, otherwise it is an empty list as a placeholder. 
            if (mesh.cells[cells_index].type == 'tetra'):
                if (iter):
                    volEleTypes.append('tetra')
                    volStartInds.append(len(tets))
                    iter = 0
                tets = np.append(tets, mesh.cells[cells_index].data, axis=0)
            elif (mesh.cells[cells_index].type == 'wedge'):
                if (iter):
                    volEleTypes.append('wedge')
                    volStartInds.append(len(wedges))
                    iter = 0
                wedges = np.append(wedges, mesh.cells[cells_index].data, axis=0)
            elif (mesh.cells[cells_index].type == 'triangle'):
                if (iter) or surfEleTypes[-1] != 'tri':
                    # If this is the first surface in the block or the prior surfaces weren't meshed with triangles, create a new block
                    surfStartInds.append(len(tris))
                    surfEleTypes.append('tri')
                    iter = 0
                tris = np.append(tris, mesh.cells[cells_index].data, axis=0)
            elif (mesh.cells[cells_index].type == 'quad'):
                if (iter) or surfEleTypes[-1] != 'quad':
                    # If this is the first surface in the block or the prior surfaces weren't meshed with quadrilaterals, create a new block
                    surfStartInds.append(len(quads))
                    surfEleTypes.append('quad')
                    iter = 0
                quads = np.append(quads, mesh.cells[cells_index].data, axis=0)
            elif (mesh.cells[cells_index].type == 'line'):
                if (iter):
                    lineStartInds.append(len(lines))
                    iter = 0
                lines = np.append(lines, mesh.cells[cells_index].data, axis=0)
            else:
                raise Exception("Unrecognized element type")

    if getBlockNames:
        return [nodes, tets, wedges, volStartInds, volEleTypes, tris, quads, surfStartInds, surfEleTypes, lines, lineStartInds, blockNames]
    else:
        return [nodes, tets, wedges, volStartInds, volEleTypes, tris, quads, surfStartInds, surfEleTypes, lines, lineStartInds]



if __name__ == "__main__":
    #filename = "test_meshio\\box_vol_surface.msh"
    filename="./Meshes/GK_surf/Mesh/meshIBM_Normal2.msh"
    [nodes_gmsh, 
    tets_gmsh, 
    wedges_gmsh, 
    volStartInds_gmsh, 
    volEleTypes_gmsh, 
    tris_gmsh, 
    surfStartInds_gmsh, 
    lines_gmsh, 
    lineStartInds_gmsh] = readGmsh(filename)

    print("Nodes:")
    print(nodes_gmsh)
    print("Tets:")
    print(tets_gmsh)
    print("Wedges:")
    print(wedges_gmsh)
    print("volStartInds: ")
    print(volStartInds_gmsh)
    print("volEleTypes:")
    print(volEleTypes_gmsh)
    print("Tris: ")
    print(tris_gmsh)
    print("surfStartInds: ")
    print(surfStartInds_gmsh)
    print("Lines: ")
    print(lines_gmsh)
    print("lineStartInds: ")
    print(lineStartInds_gmsh)