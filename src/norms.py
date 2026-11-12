import numpy as np 

def frobenius_norm(A: np.ndarray) -> float:
    sum_of_squares = np.sum(np.abs(A)**2) 
    return float(np.sqrt(sum_of_squares))

def operator_2_norm(A: np.ndarray) -> float: 
    gram_matrix = A.T.conj() @ A 
    eigenvalues = np.linalg.eigvals(gram_matrix) 
    max_eigenvalue = np.max(eigenvalues)

    max_eigenvalue = max(max_eigenvalue, 0.0) 

    return float(np.sqrt(max_eigenvalue))

if __name__ == "__main__":
    A = np.array([
        [4.0, 0.0],
        [3.0, 5.0]
    ])

    f_norm = frobenius_norm(A)
    op_norm = operator_2_norm(A)

    print(f"Matrix A:\n{A}\n")
    print(f"Manual Frobenius Norm: {f_norm:.4f}")
    print(f"Numpy  Frobenius Norm: {np.linalg.norm(A, 'fro'):.4f}\n")
    
    print(f"Manual Operator 2-Norm: {op_norm:.4f}")
    print(f"Numpy  Operator 2-Norm: {np.linalg.norm(A, 2):.4f}")

