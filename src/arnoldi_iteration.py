import numpy as np 
from typing import Tuple 

def arnoldi_iteration(A: np.ndarray, b: np.ndarray, num_steps: int, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds an orthonormal basis for the Krylov subspace X_j(A, b) using the Arnoldi method with modified 
    Gram-Schmidt orthogonalization.

    Params: 
        A: square matrix of shape (n, n)
        b: Starting vector of shape (n, 1)
        num_steps: The maximum number of Krylov steps to perform (j)

    Returns: 
        Q: Orthonormal basis matrix. Shape is (n, n+1) unless early breakdown occurs.
        H: Upper Hessenberg matrix. Shape is (j+1, j) unless early breakdown occurs.
    """
    n = A.shape[0]

    # initialize the upper Hessenberg Matrix H and the basis matrix Q 
    H = np.zeros((num_steps + 1, num_steps), dtype=np.float64)
    Q = np.zeros((n, num_steps + 1), dtype=np.float64)

    # normalize the starting vector to get q1
    Q[:, 0] = b / np.linalg.norm(b) 

    for j in range(num_steps):

        # generate the new raw vector by applying A to the latest basis vector
        w = A @ Q[:, j]

        # modified GS 
        for i in range(j + 1): 
            H[i, j] = np.dot(Q[:, i].conj(), w)

            # IMMEDIATELY subtract the projection from w before the next iteration 

            w = w - H[i, j] * Q[:, i] 
        
        # calculate the magnitude of the orthogonalized remaider
        H[j+1, j] = np.linalg.norm(w) 

        # check for exact break down(incase w is zero vector)
        if H[j+1,j] < tol: 
            print(f"--- Exact breakdown hit at step {j + 1} (h_{j+2},{j+1} = 0) ---")
            # If breakdown occurs, the Krylov space is perfectly invariant under A.
            # We truncate Q and H to remove the empty unused columns/rows.
            return Q[:, :j+1], H[:j+1, :j+1]
        # normalize to create the next orthonormal basis vector
        Q[:, j+1] = w / H[j+1,j]

    return Q, H

if __name__ == "__main__":
    A = np.eye(4) + np.ones((4, 4)) 

    b = np.array([1.0, 0.0, 0.0, 1.0])

    Q, H = arnoldi_iteration(A, b, num_steps=3)

    print("\nBasis Matrix Q (Columns are q_1, q_2):")
    print(np.round(Q, 4))
    
    print("\nHessenberg Matrix H:")
    print(np.round(H, 4))

    # Verification: Check the trace and determinant of H_2 as specified in the guide
    # Trace must be 2*alpha + 4*beta = 2(1) + 4(1) = 6
    # Det must be alpha*(alpha + 4*beta) = 1*(1 + 4) = 5
    print("\nVerification Checks:")
    print(f"Trace of H: {np.trace(H):.1f} (Expected: 6.0)")
    print(f"Det of H:   {np.linalg.det(H):.1f} (Expected: 5.0)")
    
    # Verify the Arnoldi relation holds exactly at breakdown: AQ_j = Q_j H_j
    print("\nDoes A @ Q == Q @ H hold exactly?")
    print(np.allclose(A @ Q, Q @ H))
