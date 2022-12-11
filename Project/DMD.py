# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:06:26 2022

"@author: William
"""

import numpy as np

#def DMD(W, ft, r=None):

"""    
X = np.zeros((2,10))
X[0,:] = np.arange(1,11)
X[1,:] = np.arange(2,12)
W = X
r = 2


Wminus = W[:, :-1]
Wplus = W[:, 1:]


# Comput lowcost SVD decomposition
U,S,V = np.linalg.svd(Wminus, full_matrices=False)
V = V.T

#calculate r
if r is not None:
    r = 0
    for r in range(1,S.shape[0]):
        if S[r-1,r-1]/S[r,r] > ft:
            break
#Build reduced koopman operator (3)
S_r_inv = np.reciprocal(S[:r])
A = U[:r].T@Wplus@V[:r]@S_r_inv

# Perform eigendecomposition of the reduced 
# Koopman operator with (4)
Lambda,Y = np.linalg.eig(A)

#compute the matrix of weights modes
psi = Wplus @ V[:r] @ np.diag(S_r_inv) @ Y

w = psi @ np.diag(Lambda) @ np.linalg.pinv(psi)

   
#return A, Lambda, w
"""
def DMD(data, r):
    """Dynamic Mode Decomposition (DMD) algorithm."""
    
    ## Build data matrices
    X1 = data[:, : -1]
    X2 = data[:, 1 :]
    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices = False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    
    return A_tilde, Phi, A

def DMD4cast(data, r, pred_step):
    N, T = data.shape
    _, _, A = DMD(data, r)
    mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
    for s in range(pred_step):
        mat[:, T + s] = (A @ mat[:, T + s - 1]).real
    return mat[:, - pred_step :]