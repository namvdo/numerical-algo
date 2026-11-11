import numpy as np 

def classic_gram_schmidt(A: np.ndarray): 
    """
    Computes QR factorization of matrix A using classical Gram-Schmidt.

    Returns:
        Q: Orthogonal matrix (m x n)
        R: Upper triangular matrix (n x n)
    """

    m, n = A.shape 

    Q = np.zeros((m,n),dtype=np.float64)
    R = np.zeros((n,n), dtype=np.float64)

    for j in range(n):
        # start with the j-th column of A
        v = A[:, j].astype(np.float64)

        # project v on all previously computed columns of Q (indices from 0 to j-1)
        for i in range(j):
            # r_ij = q_i dot a_j 
            R[i, j] = np.dot(Q[:, i], A[:, j])

            # subtract the projection 
            v = v - R[i, j] * Q[:, i]
        
        # the diagonal element r_jj is the magnitude of the orthogonalized vector
        R[j,j] = np.linalg.norm(v) 

        # prevent divide by zero in case of linear dependent columns
        if R[j,j] > 1e-12:
            Q[:,j] = v / R[j, j]
        else:
            Q[:,j] = v 

    return Q, R 

if __name__ == "__main__":
    A = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ])

    Q, R = classic_gram_schmidt(A) 
    print("Matrix Q (Orthogonal basis):\n", np.round(Q, 4))
    print("\nMatrix R (Upper triangular):\n", np.round(R, 4))

    print("Verification Q @ R = A")
    print(np.allclose(Q @ R, A))

    I = np.eye(A.shape[1])
    print("\nVerify Q^T @ Q = I")
    print(np.allclose(Q.T @ Q, I))
