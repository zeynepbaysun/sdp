import scipy as sp
import numpy as np
import gzip
import os

def read_sparse_matrix_from_gz(file_name):
    # Get the absolute path to the file in the project directory
    file_path = os.path.join(os.getcwd(), file_name)

    with gzip.open(file_path, 'rt') as f:
        # Read the decompressed file using mmread
        matrix = sp.io.mmread(f)
    return matrix

# Convert sparse matrix to CSR format
def convert_to_csr(matrix):
    return matrix.tocsr()


# read into sparse matrix:
file_name = '1138_bus.mtx.gz'  # Replace 'your_file_name.gz' with the actual file name
sparse_matrix = read_sparse_matrix_from_gz(file_name)
print(sparse_matrix)

# Convert to CSR format
csr_matrix = convert_to_csr(sparse_matrix)
print(csr_matrix)
