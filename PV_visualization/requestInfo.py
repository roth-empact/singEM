# For timestepping, see https://docs.paraview.org/en/latest/ReferenceManual/pythonProgrammableFilter.html#reading-a-csv-file-series for details
def setOutputTimesteps(algorithm, timesteps):
    "helper routine to set timestep information"
    executive = algorithm.GetExecutive()
    outInfo = executive.GetOutputInformation(0)

    outInfo.Remove(executive.TIME_STEPS())
    for timestep in timesteps:
        outInfo.Append(executive.TIME_STEPS(), timestep)

    outInfo.Remove(executive.TIME_RANGE())
    outInfo.Append(executive.TIME_RANGE(), timesteps[0])
    outInfo.Append(executive.TIME_RANGE(), timesteps[-1])

setOutputTimesteps(self, range(int("NUM_TIMESTEPS"))) # Flag will be replaced by an integer value in sourceLoader.py

