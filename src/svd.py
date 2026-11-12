import numpy as np 
from typing import Tuple 

def solve_svd(A: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> np.ndarray: 
    """
    Solve Ax = b using SVD 
    """

    U, S, Vt = svd(A) 
    V = Vt.T 

    n = len(S) 
    y = np.zeros(n, dtype=np.float64) 

    for i in range(n): 
        if S[i] > tol: 
            c_i = np.dot(U[:, i], b) 
            y[i] = c_i / S[i] 
        else: 
            y[i] = 0.0 
    
    x = np.dot(V, y) 
    return x 



def svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Computes the SVD (A = U * Sigma * V^T) following the Gram-matrix method 

    Parameters: 
        A: 2D numpy.ndarray of shape (m, n)

    Returns: 
        U: 2D numpy.ndarray (Left singular vectors)
        S: 1D numpy.ndarray (Singular values in descending order)
        Vt: 2D numpy.ndarray (Tranpose of right singular vectors) 
    """
    m, n = A.shape 

    gram_matrix = A.T @ A 

    eigenvalues, V = np.linalg.eigh(gram_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    S = np.sqrt(eigenvalues)

    U = np.zeros((m,n), dtype=np.float64)

    for i in range(n):
        if S[i] > 1e-12: 
            U[:, i] = (A @ V[:, i]) / S[i]
        else:
            U[:, i] = np.zeros(m) 

    return U, S, V.T 

if __name__ == "__main__":
    A = np.array([
        [4.0, 0.0],
        [3.0, 5.0]
    ])

    U, S, Vt = svd(A)

    print("Matrix U (Left singular vectors):\n", np.round(U, 4))
    print("\nSingular Values (Sigma):\n", np.round(S, 4))
    print("\nMatrix V^T (Right singular vectors transposed):\n", np.round(Vt, 4))

    Sigma_matrix = np.diag(S) 
    A_reconstructed = U @ Sigma_matrix @ Vt 

    print("\nVerification (U @ Sigma @ V^T == A):")
    print(np.allclose(A, A_reconstructed))
    print("A_reconstructed: ", A_reconstructed)


    A = np.array([
        [4.0, 0.0],
        [3.0, 5.0]
    ])
    b = np.array([4.0, 8.0])

    x = solve_svd(A, b) 

    print("Computed solution: ", x)
    print("Verification A @ x = b", A @ x )




