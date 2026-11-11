import numpy as np 

def back_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray: 
    """
    Solve upper triangular system Ux = y

    Params: 
    U: 2D numpy.ndarray of shape (n, n) represent the upper triangular matrix.
    y: 1D numpy.ndarray of shape (n,) representing the rhs vector.

    Returns:
    x: 1D numpy.ndarray of shape (n, ) representing the solution vector.
    """

    n = len(y)

    x = np.zeros_like(y, dtype=np.float64)

    for i in range(n-1, -1, -1):
        if np.isclose(U[i, i], 0.0):
            raise ZeroDivisionError("Matrix is singular")
        
        known_sum = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - known_sum) / U[i, i]
    return x 

if __name__ == "__main__":
    U = np.array([
        [2.0, 1.0, -1.0],
        [0.0, 3.0, 4.0],
        [0.0, 0.0, -2.0]
    ])

    y = np.array([8.0, 5.0, 2.0])

    x = back_substitution(U, y)

    print("Computed solution x: ", x)

    print("Verfication (U @ x ) = y", U @ x )
    res = U @ x 

    assert np.allclose(res, y)
