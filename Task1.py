import numpy as np

def compute_eigen(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    print(f"Eigenvalues: {eigenvalues}")
    print(f"\nEigenvectors: {eigenvectors}")

    print("\nIs A·v = λ·v?")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        λ = eigenvalues[i]
        left = A @ v
        right = λ * v
        print(f"Vector {i + 1}: ", np.allclose(left, right))
    return eigenvalues, eigenvectors

A = np.array([
    [2, 1],
    [1, 2]
])

compute_eigen(A)
