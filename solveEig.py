import numpy as np
import scipy.sparse as spa
import time
from linalg_funcs import eigs_SLEPc, matrix_remap, match_rank

# A function to perform eigenvalue / eigenvector extraction using the provided matrices
def solveEig(matStruct, numEig, solveP, simType, numProcs):
  K1 = matStruct["K1"]
  K2 = matStruct["K2"]
  Tee = matStruct["Tee"]
  See = matStruct["See"]
  if simType == "A" or simType == "TCS":
    Ken = matStruct["Ken"]
    Kne = matStruct["Kne"]
    Gnn = matStruct["Gnn"]
    Ten = matStruct["Ten"]
    Sen = matStruct["Sen"]
    if simType == "TCS":
      Tne = matStruct["Tne"]
      Sne = matStruct["Sne"]
      Tnn = matStruct["Tnn"]
      Snn = matStruct["Snn"]

  ep0 = 1e-9 / (36 * np.pi)
  mu0 = 4e-7 * np.pi

  sigma = (2*np.pi*solveP["shift"])**2

  if simType == "E":
    A = K1 + See
    B = K2 + Tee
    # totMat = A - solveP["sigma"]*B
    # print("Building preconditioner")
    # start_time = time.time()
    # M_diag = spa.diags_array(1/np.sqrt(np.abs(totMat.diagonal())), format="csr")
    # totMat_ilu = spa.linalg.spilu((M_diag @ totMat @ M_diag).tocsc(), drop_tol=solveP["dropTol"], fill_factor=((solveP["rowFill"]/100)*A.shape[0]**2/A.nnz))
    # M_ilu = spa.linalg.LinearOperator(A.shape, totMat_ilu.solve)
    # print(f"Preconditioning done. Time elapsed: {(time.time() - start_time):.3f} s. Sparsity percentage: {(100*(1-(totMat_ilu.L.nnz + totMat_ilu.U.nnz)/(N_e+N_n)**2)):.3f}%") 
    # def totMat_func(b):
    #   return M_diag @ (spa.linalg.gmres(M_diag @ totMat @ M_diag, M_diag @ b, restart=solveP["restart"], 
    #                                          maxiter=solveP["maxit"], rtol=solveP["tol"], M=M_ilu)[0])
    # totMat_linop = spa.linalg.LinearOperator(A.shape, matvec=totMat_func)

    if solveP["pkg"] == "SciPy":
      start_time = time.time()
      print("Finding eigenvalues")
      if numEig < 20:
        eigVals, eigVecs = spa.linalg.eigs(A, M=B, which = 'LM', k = 20, sigma=sigma, tol=solveP["tol"])
        # import scipy as sc
        # eigVals, eigVecs = sc.linalg.eig(A.todense(), b=B.todense())
      else:
        eigVals, eigVecs = spa.linalg.eigs(A, M=B, which = 'LM', k = numEig, sigma=sigma, tol=solveP["tol"])
      posEigvals = eigVals > sigma/1e3
      eigVals = eigVals[posEigvals]
      eigVecs = eigVecs[:,posEigvals]
    elif solveP["pkg"] == "SLEPc":
      eigVals, eigVecs = eigs_SLEPc(A, B, numProcs, numEig, sigma)

  elif simType == "A":
    K1_coeff = mu0
    Ken_coeff = 1/ep0
    Kne_coeff = mu0*ep0
    Gnn_coeff = 1
    if See.nnz == 0:
      A = spa.block_array([[K1_coeff*(K1), -Ken_coeff*(Ken+Ten)], 
                                 [Kne_coeff*Kne, -Gnn_coeff*Gnn]])
      B = spa.block_array([[K1_coeff*(K2+Tee), None],
                                 [None, spa.csr_array(Gnn.shape, dtype=np.float64)]])
      M_diag = spa.diags_array(1/np.sqrt(np.abs(A.diagonal())), format="csr")
    else: # Form an enlarged system with a dummy variable if inductances are used
      # Shrink the column dimension of Sen to only nodes on relevant ports
      Sen_cond, P_Sen = matrix_remap(Sen, 1) 
      Sen_colNorms = np.zeros((Sen_cond.shape[1],)) # Discard columns with roughly zero magnitude
      for ii in range(Sen_cond.shape[1]):
        Sen_colNorms[ii] = np.linalg.norm(Sen_cond[:,[ii]].todense())
      maxNorm = np.max(Sen_colNorms)
      nzCols = np.where(Sen_colNorms/maxNorm > 1e-6)[0]
      Sen_cond = Sen_cond[:,nzCols]
      P_Sen = P_Sen[nzCols,:]

      if P_Sen.shape[0] < 1000: ## Only perform rank reduction if lumped ports are used
                                #  TODO: Implement a better way of distinguishing this than using the new rank
        # Check if the remaining columns are linear combinations of each other - if yes, remove them
        Sen_cond, scalMat, matRank = match_rank(Sen_cond, 1e-6)
        # Adjust the permutation matrix to reflect the removed rows
        ind = 0
        P_Sen_red = spa.lil_array((matRank, Gnn.shape[0]))
        for ii in range(len(scalMat)):
          if scalMat[ii,ii] == 1: # Treat this as the independent column
            P_Sen_red[[ind],:] += P_Sen[[ii],:]
            for jj in range(ii+1,len(scalMat)):
              P_Sen_red[[ind],:] += scalMat[ii,jj]*P_Sen[[jj],:]
            ind += 1
        P_Sen_red = P_Sen_red.asformat("csr")
      else:
        P_Sen_red = P_Sen

      # Now form the full block system
      eq1_scal = mu0
      eq2_scal = mu0*ep0
      eq3_scal = mu0*ep0
      var1_scal = 1
      var2_scal = 1/(mu0*ep0)
      var3_scal = Gnn.shape[0]*var2_scal*spa.linalg.norm(Ken)/spa.linalg.norm(Sen_cond)
      # var3_scal = var2_scal*(spa.linalg.norm(Ken)/(spa.linalg.norm(Sen_cond)))*(spa.linalg.norm(See)/spa.linalg.norm(K1)) # For scaling the dummy variable
      scalMat = np.outer(np.array([eq1_scal, eq2_scal, eq3_scal]), np.array([var1_scal, var2_scal, var3_scal]))

      A = spa.block_array([[scalMat[0,0]*(K1+See), -scalMat[0,1]*(Ken+Ten), scalMat[0,2]*Sen_cond], 
                                    [scalMat[1,0]*Kne, -scalMat[1,1]*Gnn, None],
                                    [None, scalMat[2,1]*P_Sen_red, None]], format='csr')
      B = spa.block_array([[scalMat[0,0]*(K2+Tee), None, None],
                                 [None, spa.csr_array(Gnn.shape, dtype=np.float64), None],
                                 [None, None, scalMat[2,2]*spa.eye_array(Sen_cond.shape[1])]], format='csr')
      A_diag_mod = A.diagonal()
      A_diag_mod[K1.shape[0]+Gnn.shape[0]:] = 1/(P_Sen_red @ Gnn @ P_Sen_red.T).diagonal()
      M_diag = spa.diags_array(1/np.sqrt(np.abs(A_diag_mod)), format="csr")
    start_time = time.time()
    if solveP["pkg"] == "SciPy":
      print("Building preconditioner")
      
      A_ilu = spa.linalg.spilu((M_diag @ A @ M_diag).tocsc(), drop_tol=solveP["dropTol"], fill_factor=solveP["rowFill"])
      M_ilu = spa.linalg.LinearOperator(A.shape, A_ilu.solve)
      print(f"Preconditioning done. Time elapsed: {(time.time() - start_time):.3f} s. Sparsity percentage: {(100*(1-(A_ilu.L.nnz + A_ilu.U.nnz)/A.shape[0]**2)):.3f}%") 
      def A_func(b):
        x = spa.linalg.gmres(M_diag @ A @ M_diag, M_diag @ b, restart=solveP["restart"], 
                                              maxiter=solveP["maxit"], rtol=solveP["tol"], M=M_ilu)
        # x[0][N_e:] = x[0][N_e:]*K1_coeff
        # x[0][:N_e] = x[0][:N_e]*Ken_coeff
        if x[1] != 0:
          print(f"Iterative solver did not converge. Relative residual: {np.linalg.norm(A.dot(M_diag @ x[0]) - b) / np.linalg.norm(b)}")
        return M_diag @ x[0]
      A_linop = spa.linalg.LinearOperator(A.shape, matvec=A_func)

      start_time = time.time()
      print("Finding eigenvalues")
      if numEig < 20:
        eigVals, eigVecs = spa.linalg.eigs(B, M=A, Minv=A_linop, which = 'LM', k = 20, tol=solveP["tol"])
      else:
        eigVals, eigVecs = spa.linalg.eigs(B, M=A, Minv=A_linop, which = 'LM', k = numEig, tol=solveP["tol"])
      
    elif solveP["pkg"] == "SLEPc":
      if B.format != 'csr':
        B = B.asformat('csr')
      eigVals, eigVecs = eigs_SLEPc(B, A, numProcs, numEig, 0.0, "LM")
    eigVals = np.sort(1/eigVals)

    if See.nnz != 0: # If inductances are used, discard the dummy variable
      eigVals = eigVals[:K1.shape[0]+Gnn.shape[0]]
      eigVecs = eigVecs[:K1.shape[0]+Gnn.shape[0], :K1.shape[0]+Gnn.shape[0]]

    # eigVecs[N_e:, :] = eigVecs[N_e:, :]*K1_coeff
    # eigVecs[:N_e, :] = eigVecs[:N_e, :]*Ken_coeff

  elif simType == "TCS":
    A = spa.block_array([[K1+See, Sen], 
                               [Sne, Snn]])
    B = spa.block_array([[K2+Tee, Ken+Ten],
                               [Kne+Tne, Gnn+Tnn]])
    print("Building preconditioner")
    start_time = time.time()
    M_diag = spa.diags_array(1/np.sqrt(np.abs(B.diagonal())), format="csr")
    B_ilu = spa.linalg.spilu((M_diag @ B @ M_diag).tocsc(), drop_tol=solveP["dropTol"], fill_factor=solveP["rowFill"])
    M_ilu = spa.linalg.LinearOperator(B.shape, B_ilu.solve)
    print(f"Preconditioning done. Time elapsed: {(time.time() - start_time):.3f} s. Sparsity percentage: {(100*(1-(B_ilu.L.nnz + B_ilu.U.nnz)/(A.shape[0])**2)):.3f}%") 
    def B_func(b):
      x = spa.linalg.gmres(M_diag @ B @ M_diag, M_diag @ b, restart=solveP["restart"], 
                                             maxiter=solveP["maxit"], rtol=solveP["tol"], M=M_ilu)
      if x[1] != 0:
        print(f"Iterative solver did not converge. Relative residual: {np.linalg.norm(B.dot(M_diag @ x[0]) - b) / np.linalg.norm(b)}")
      return M_diag @ x[0]
    B_linop = spa.linalg.LinearOperator(B.shape, matvec=B_func)

    start_time = time.time()
    print("Finding eigenvalues")
    if numEig < 20:
      eigVals, eigVecs = spa.linalg.eigs(A, M=B, which = 'LM', k = 20, sigma=sigma, tol=solveP["tol"])
      # eigVals, eigVecs = spa.linalg.eigs(A, M=B, Minv=B_linop, which = 'SM', k = 20, tol=solveP["tol"])
    else:
      eigVals, eigVecs = spa.linalg.eigs(A, M=B, which = 'LM', k = numEig, sigma=sigma, tol=solveP["tol"])
      # eigVals, eigVecs = spa.linalg.eigs(A, M=B, Minv=B_linop, which = 'SM', k = numEig, tol=solveP["tol"])

    sort = np.argsort(eigVals)
    eigVals = eigVals[sort[:numEig]]
    eigVecs = eigVecs[:,sort[:numEig]]

  if len(eigVals) > numEig:
    eigVals = eigVals[:numEig]
    eigVecs = eigVecs[:,:numEig]
  
  return eigVals, eigVecs