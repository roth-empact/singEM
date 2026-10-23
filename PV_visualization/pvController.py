import subprocess

# write temporary args file for interfacing with the paraview-run sourceLoader

def setArgs(args):
    with open("args", 'w') as file:
        for arg in args:
            file.write(arg + ",")

# launching paraview gui
## [Hardcoded] Paths to pv executables
PV_PATH       = r"C:\Users\selkin\Applications\ParaView-5.12.0-Windows-Python3.10-msvc2017-AMD64\bin\paraview.exe"
PVPYTHON_PATH = r"C:\Users\selkin\Applications\ParaView-5.12.0-Windows-Python3.10-msvc2017-AMD64\bin\pvpython.exe"

def launchPV(args):
    setArgs(args)
    subprocess.run([PV_PATH, "--script=./PV_visualization/sourceLoader.py"])

# launching pvpython

def launchPVPython(args):
    setArgs(args)
    subprocess.run([PVPYTHON_PATH, "--script=./PV_visualization/sourceLoader.py"])

# testing

if __name__ == '__main__':
    meshfile = r"C:\Users\selkin\OneDrive - purdue.edu\Documents\Research\Code\A-phi-Testing\Meshes\microstrip.inp"
    launchPV(['./microstrip.inp', r"./E_data.npy", 10, False])
    #launchPVPython(['test2.inp', 'test.npy'])
    #launchPV(['test2.inp', 'test.npy'])