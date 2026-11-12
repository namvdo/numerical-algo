import numpy as np 
from scipy.linalg import null_space 
from typing import Tuple 


def factor_into_symmetric_matrices(A: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    num_vars = n * (n + 1) // 2 
    num_eqs = n * (n -1 ) //2 

    def var_idx(i: int, j: int) -> int: 
        if i > j: 
            i, j = j, i 
        return i * n - i * (i + 1) // 2 + j 
    
    M = np.zeros((num_eqs, num_vars), dtype=np.float64) 
    eq_idx = 0
    for p in range(n): 
        for q in range(p+1, n):
            for k in range(n):
                # enforce AS_pq - AS_qp = 0
                # term from AS_pq = sum_k A_pk S_kq 
                idx1 = var_idx(k, q) 
                M[eq_idx, idx1] += A[p, k] 

                # term from -AS_qp = -sum_k A_qk S_kp
                idx2 = var_idx(k,p)
                M[eq_idx, idx2] -= A[q, k]
            eq_idx += 1 
    
    Z = null_space(M)

    if Z.shape[1] == 0: 
        raise ValueError("No symmetric S satisfies AS = SA^T.")
    
    # must choose a solution where det(S) != 0

    np.random.seed(42) 
    S = np.zeros((n, n), dtype=np.float64)

    for _ in range(20): 
        coeffs = np.random.randn(Z.shape[1])
        v = Z @ coeffs 

        for i in range(n):
            for j in range(i, n):
                S[i, j] = S[j, i] = v[var_idx(i,j)]

        if abs(np.linalg.det(S)) > tol: 
            S1 = A @ S 
            S2 = np.linalg.inv(S) 

            S1 = (S1 + S1.T)/2.0
            S2 = (S2 + S2.T)/2.0

            return S1, S2
    raise RuntimeError("Failed to find an invertable S within the nullspace")

if __name__ == "__main__":
    A = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ])

    S1, S2 = factor_into_symmetric_matrices(A)
    print("Factor S1 (Symmetric):\n", np.round(S1, 4))
    print("\nFactor S2 (Symmetric):\n", np.round(S2, 4))
    
    print("\nVerification (S1 @ S2 == A):")
    print(np.allclose(S1 @ S2, A))
    
    print("Verification (S1 is symmetric):")
    print(np.allclose(S1, S1.T))
    
    print("Verification (S2 is symmetric):")
    print(np.allclose(S2, S2.T))

