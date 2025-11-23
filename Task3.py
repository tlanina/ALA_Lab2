import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors,np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message_(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonal_inv = np.diag(1 / eigenvalues)
    inverse_key_matrix = np.dot(np.dot(eigenvectors, diagonal_inv),np.linalg.inv(eigenvectors))
    decrypted_vector = np.dot(inverse_key_matrix, encrypted_vector)
    decrypted_vector = np.real(decrypted_vector)
    decrypted_vector = np.round(decrypted_vector).astype(int)
    decrypted_message = ''.join(chr(x) for x in decrypted_vector)
    return decrypted_message
message = "Hello, World!"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))

enc = encrypt_message(message, key_matrix)
dec = decrypt_message_(enc, key_matrix)

print(f"Original Message: {message}")
print(f"Encrypted Message: {enc}")
print(f"Decrypted Message: {dec}")
