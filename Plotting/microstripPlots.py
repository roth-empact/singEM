import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Get FEM S-parameters and frequencies
results = np.load(f"./Results/microstrip.npz", allow_pickle=True)
freqs = results['freqs']
portS = results['portS']

# Plot S11 and S21
fig1, ax1 = plt.subplots()
ax1.plot(freqs/1e9, 10*np.log10(np.abs(portS[:,0])), '-k', linewidth = 1)
ax1.grid()
ax1.set_xlabel(r"$f$ (GHz)", fontfamily='Times New Roman', fontsize=14)
ax1.set_ylabel(r"$S_{11}$ (dB)", fontfamily='Times New Roman', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12, labelfontfamily='Times New Roman')
fig1.savefig("./Results/microstrip_S11.png", pad_inches=0.01)

fig2, ax2 = plt.subplots()
ax2.plot(freqs/1e9, 10*np.log10(np.abs(portS[:,1])), '-k', linewidth = 1)
ax2.grid()
ax2.set_xlabel(r"$f$ (GHz)", fontfamily='Times New Roman', fontsize=14)
ax2.set_ylabel(r"$S_{21}$ (dB)", fontfamily='Times New Roman', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12, labelfontfamily='Times New Roman')
fig2.savefig("./Results/microstrip_S21.png", pad_inches=0.01)

plt.show()