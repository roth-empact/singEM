#!python
cubit.cmd('reset')
import numpy
import cubit

############## Assemble some string definitions that will be used repeatedly
genBrick = "brick x {} y {} z {}"
genRectY = "create surface rectangle width {} height {} yplane"
genRectZ = "create surface rectangle width {} height {} zplane"
genLine = "create curve location {} {} {} location {} {} {}"
subSurf = "subtract surface {} from surface {}"
moveVol = "move volume {} x {} y {} z {}"
moveSurf = "move surface {} x {} y {} z {}"
imprintVol = "imprint volume {} {}"
mergeVol = "merge volume {} {}"
blockVol = "block {} add volume {}"
blockSurf = "block {} add surf {}"
blockLine = "block {} add curve {}"
nameBlock = "block {} name \"{}\""

############## Geometry definitions
# Substrate
sub_width = 5e-3
sub_height = 8e-4
sub_length = 10e-3
sub_er = 7

# Signal line
sig_width = sub_width / 5
#sig_height = sub_height / 20
sig_length = sub_length

# Ground plane
gnd_width = sub_width
#gnd_height = sig_height*3
gnd_length = sub_length

# Bounding box
box_width = sub_width*2
box_height = sub_height*10
box_length = sub_length*2

# Lumped ports
port_width = sig_width
port_height = sub_height

############## Cubit calls
# Create substrate volume and find the surfaces where ports will lie
cubit.cmd(genBrick.format(sub_width, sub_height, sub_length))
sub_vol = cubit.get_last_id("volume")
sub_surfs = cubit.parse_cubit_list("surface", f"in volume {sub_vol}")
sub_surf_port1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3}")
sub_surf_port2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-1*sub_length/3}")

# # Create signal conductor volume and move to lie atop the substrate
# cubit.cmd(genBrick.format(sig_width, sig_height, sig_length))
# sig_vol = cubit.get_last_id("volume")
# cubit.cmd(moveVol.format(sig_vol, 0, sub_height/2 + sig_height/2, 0))
# sig_vol = cubit.get_last_id("volume")

# Create signal conductor surface
cubit.cmd(genRectY.format(sig_width, sig_length))
sig_vol = cubit.get_last_id("volume")
cubit.cmd(moveVol.format(sig_vol, 0, sub_height/2, 0))

# # Create ground conductor volume
# cubit.cmd(genRectZ.format(gnd_width, gnd_height, gnd_length))
# gnd_vol = cubit.get_last_id("volume")
# cubit.cmd(moveVol.format(gnd_vol, 0, -sub_height/2 - gnd_height/2, 0))

# Create ground conductor surface
cubit.cmd(genRectY.format(gnd_width, gnd_length))
gnd_vol = cubit.get_last_id("volume")
cubit.cmd(moveVol.format(gnd_vol, 0, -sub_height/2, 0))

# Create bounding box
cubit.cmd(genBrick.format(box_width, box_height, box_length))
box_vol = cubit.get_last_id("volume")

# Create lumped ports
cubit.cmd(genRectZ.format(port_width, port_height))
port1_vol = cubit.get_last_id("volume")
cubit.cmd(moveVol.format(port1_vol, 0, 0, sub_length/2))

cubit.cmd(genRectZ.format(port_width, port_height))
port2_vol = cubit.get_last_id("volume")
cubit.cmd(moveVol.format(port2_vol, 0, 0, -sub_length/2))

# Imprint and merge surfaces
cubit.cmd("imprint all")
cubit.cmd("merge all")

# Delete volumes for lumped ports (surfaces have already been merged)
port1_surf = cubit.parse_cubit_list("surface", f"in volume {port1_vol}")[0]
port2_surf = cubit.parse_cubit_list("surface", f"in volume {port2_vol}")[0]
cubit.cmd(f"delete body {port1_vol}")
cubit.cmd(f"delete body {port2_vol}")

# Delete volume for signal conductor
sig_surf = cubit.parse_cubit_list("surface", f"in volume {sig_vol}")[0]
cubit.cmd(f"delete body {sig_vol}")
print(sig_surf)

# Delete volume for GND
gnd_surf = cubit.parse_cubit_list("surface", f"in volume {gnd_vol}")[0]
cubit.cmd(f"delete body {gnd_vol}")
print(gnd_surf)

# Split the lumped port surfaces down a center line
cubit.cmd(f"split surface {port1_surf} across location position {0} {sub_height/2} {sub_length/2} location position {0} {-sub_height/2} {sub_length/2}")
port1_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
port1_surf2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord<{0} with x_coord>{-sub_width/4}")[0]
port1_line = cubit.parse_cubit_list("curve", f"in surface {port1_surf1} with x_coord={0}")[0]

cubit.cmd(f"split surface {port2_surf} across location position {0} {sub_height/2} {-sub_length/2} location position {0} {-sub_height/2} {-sub_length/2}")
port2_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
port2_surf2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord<{0} with x_coord>{-sub_width/4}")[0]
port2_line = cubit.parse_cubit_list("curve", f"in surface {port2_surf1} with x_coord={0}")[0]

# Remove center region from box (to allow meshing to function properly)
cubit.cmd(f"subtract volume {sub_vol} from volume {box_vol} keep_tool") # Keep_tool avoids removing the subtracted volumes

# Imprint and merge surfaces
cubit.cmd("imprint all")
cubit.cmd("merge all")

# Need to do it twice for some reason
cubit.cmd("imprint all")
cubit.cmd("merge all")

# Compress labels and find new box volume ID
cubit.cmd("compress")
box_vol = cubit.get_last_id("volume")

## Mesh volumes
# # Start by meshing along the port integration lines to ensure they're dense enough
# port1_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
# port1_line = cubit.parse_cubit_list("curve", f"in surface {port1_surf1} with x_coord={0}")[0]
# cubit.cmd(f"curve {port1_line} interval 20")
# cubit.cmd(f"curve {port1_line} scheme equal")
# cubit.cmd(f"mesh curve {port1_line}")

# port2_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
# port2_line = cubit.parse_cubit_list("curve", f"in surface {port2_surf1} with x_coord={0}")[0]
# cubit.cmd(f"curve {port2_line} interval 20")
# cubit.cmd(f"curve {port2_line} scheme equal")
# cubit.cmd(f"mesh curve {port2_line}")

# Start by meshing the port surfaces to ensure they're dense enough
port1_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
port1_surf2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord<{0} with x_coord>{-sub_width/4}")[0]
cubit.cmd(f"surface {port1_surf1} {port1_surf2} size auto factor 9")
cubit.cmd(f"surface {port1_surf1} {port1_surf2} scheme trimesh")
cubit.cmd(f"mesh surface {port1_surf1} {port1_surf2}")

port2_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
port2_surf2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord<{0} with x_coord>{-sub_width/4}")[0]
cubit.cmd(f"surface {port2_surf1} {port2_surf2} size auto factor 9")
cubit.cmd(f"surface {port2_surf1} {port2_surf2} scheme trimesh")
cubit.cmd(f"mesh surface {port2_surf1} {port2_surf2}")

# Now mesh the remainder of the substrate volume
cubit.cmd(f"volume {sub_vol} size auto factor 7")
cubit.cmd(f"volume {sub_vol} scheme tetmesh")
cubit.cmd(f"mesh volume {sub_vol}")

# # Mesh PEC surfaces
# sig_surf = cubit.parse_cubit_list("surface", f"in volume {sig_vol} with y_coord>{sub_height/2}")
# for surfNo in sig_surf:
#     cubit.cmd(f"surface {surfNo} size auto factor 5")
#     cubit.cmd(f"surface {surfNo} scheme trimesh")
#     cubit.cmd(f"mesh surface {surfNo}")

# gnd_surf = cubit.parse_cubit_list("surface", f"in volume {gnd_vol} with y_coord<{-sub_height/2}")
# for surfNo in gnd_surf:
#     cubit.cmd(f"surface {surfNo} size auto factor 5")
#     cubit.cmd(f"surface {surfNo} scheme trimesh")
#     cubit.cmd(f"mesh surface {surfNo}")

# Mesh volume of PEC box
cubit.cmd(f"volume {box_vol} size auto factor 7")
cubit.cmd(f"volume {box_vol} scheme tetmesh")
cubit.cmd(f"mesh volume {box_vol}")

# cubit.cmd(f"volume {sub_vol} {box_vol} scheme tetmesh")
# cubit.cmd(f"mesh volume {sub_vol} {box_vol}")

## Create blocks
cubit.cmd("set duplicate block elements off")

# Blocks for each (non-PEC) volume
blockCnt = 1 # block index for the substrate
cubit.cmd(blockVol.format(blockCnt, sub_vol))
cubit.cmd(nameBlock.format(blockCnt, f"mpr1epr{sub_er}"))
blockCnt += 1 # Index 2 for the box (free space)
cubit.cmd(blockVol.format(blockCnt, box_vol))
cubit.cmd(nameBlock.format(blockCnt, "mpr1epr1"))

# A block for the external PEC surfaces
blockCnt += 1
boxSurfs = [] # List to store indices of the outer surfaces of the box
boxSurfs.append(cubit.parse_cubit_list("surface", f"in volume {box_vol} with x_coord={box_width/2}")[0])
boxSurfs.append(cubit.parse_cubit_list("surface", f"in volume {box_vol} with x_coord={-box_width/2}")[0])
boxSurfs.append(cubit.parse_cubit_list("surface", f"in volume {box_vol} with y_coord={box_height/2}")[0])
boxSurfs.append(cubit.parse_cubit_list("surface", f"in volume {box_vol} with y_coord={-box_height/2}")[0])
boxSurfs.append(cubit.parse_cubit_list("surface", f"in volume {box_vol} with z_coord={box_length/2}")[0])
boxSurfs.append(cubit.parse_cubit_list("surface", f"in volume {box_vol} with z_coord={-box_length/2}")[0])
print(boxSurfs)
for ii in boxSurfs:
    cubit.cmd(blockSurf.format(blockCnt, ii))
cubit.cmd(nameBlock.format(blockCnt, "PEC0"))

blockCnt += 1
sig_surf = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with y_coord={sub_height/2} with x_coord={0}")[0]
cubit.cmd(blockSurf.format(blockCnt, sig_surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC1"))

blockCnt += 1
gnd_surf = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with y_coord={-sub_height/2}")[0]
cubit.cmd(blockSurf.format(blockCnt, gnd_surf))
cubit.cmd(nameBlock.format(blockCnt, "PEC2"))

# A block for the input port
blockCnt += 1
port1_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
port1_surf2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord>{sub_length/3} with x_coord<{0} with x_coord>{-sub_width/4}")[0]
cubit.cmd(blockSurf.format(blockCnt, port1_surf1))
cubit.cmd(blockSurf.format(blockCnt, port1_surf2))
cubit.cmd(nameBlock.format(blockCnt, "Port1"))

# A block for the output port
blockCnt += 1
port2_surf1 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord>{0} with x_coord<{sub_width/4}")[0]
port2_surf2 = cubit.parse_cubit_list("surface", f"in volume {sub_vol} with z_coord<{-sub_length/3} with x_coord<{0} with x_coord>{-sub_width/4}")[0]
cubit.cmd(blockSurf.format(blockCnt, port2_surf1))
cubit.cmd(blockSurf.format(blockCnt, port2_surf2))
cubit.cmd(nameBlock.format(blockCnt, "Port2"))

# # A block for the input port line
# blockCnt += 1
# port1_line = cubit.parse_cubit_list("curve", f"in surface {port1_surf1} with x_coord={0}")[0]
# cubit.cmd(blockLine.format(blockCnt, port1_line))
# cubit.cmd(nameBlock.format(blockCnt, "pLine1"))

# # A block for the output port line
# blockCnt += 1
# port2_line = cubit.parse_cubit_list("curve", f"in surface {port2_surf1} with x_coord={0}")[0]
# cubit.cmd(blockLine.format(blockCnt, port2_line))
# cubit.cmd(nameBlock.format(blockCnt, "pLine2"))


## Export the mesh
# cubit.cmd("export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code//CQEM-FEM/3D_Potential_FEM/Meshes/Microstrips/microstrip.inp\" mesh_only overwrite everything")



