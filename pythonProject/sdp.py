import scipy as sp
import numpy as np
import gzip
import tarfile
import os


def read_sparse_matrix_from_gz(file_name):
    # Get the absolute path to the file in the project directory
    file_path = os.path.join(os.getcwd(), file_name)

    with gzip.open(file_path, 'rt') as f:
        # Read the decompressed file using mmread
        matrix = sp.io.mmread(f)
        csr = matrix.tocsr()
        return csr



# Function to extract all .mtx files from a tar.gz archive
def extract_mtx_files_from_tar_gz(archive_path):
    mtx_files = []
    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar.getmembers():
            # Check if the member is a file and ends with .mtx
            if member.isfile() and member.name.endswith('.mtx'):
                # Extract the file to the current working directory
                tar.extract(member)
                mtx_files.append(member.name)
    return mtx_files


def read_sparse_matrix_csr(file_path):
    with open(file_path, 'r') as file:
        # Read header to determine matrix dimensions and number of non-zero entries
        nrows, ncols, nnz = None, None, None
        for line in file:
            if line.startswith('%'):
                continue
            else:
                # Extract dimensions and non-zero count
                parts = line.strip().split()
                nrows = int(parts[0])
                ncols = int(parts[1])
                nnz = int(float(parts[2]))  # Convert floating-point count to integer
                break

        if nrows is None or ncols is None or nnz is None:
            raise ValueError("Invalid matrix header")

        # Initialize arrays to store data for CSR matrix
        rows = []
        cols = []
        data = []

        # Read and process matrix data
        for line in file:
            if not line.startswith('%'):
                row, col, value = map(float, line.strip().split())  # Allow floating-point values
                rows.append(int(row) - 1)  # Adjust to 0-based indexing
                cols.append(int(col) - 1)  # Adjust to 0-based indexing
                data.append(value)

        # Create CSR matrix
        csr_matrix_data = sp.sparse.csr_matrix((data, (rows, cols)), shape=(nrows, ncols))
        return csr_matrix_data

# Example usage
if __name__ == "__main__":
    # Example path to tar.gz archive containing the sparse matrix files
    archive_path = "1138_bus.mtx.gz"
    if ".tar." in archive_path:
        # Extract all .mtx files from the archive
        mtx_files = extract_mtx_files_from_tar_gz(archive_path)

        # Read and print each sparse matrix in CSR format
        for mtx_file in mtx_files:
            print(f"Processing file: {mtx_file}")
            csr_matrix_data = read_sparse_matrix_csr(mtx_file)
            # Print CSR matrix format
            print("CSR Matrix Data:")
            print(csr_matrix_data)

    elif ".mtx." in archive_path:
        csr_matrix_data = read_sparse_matrix_from_gz(archive_path)
        # Print CSR matrix format
        print("CSR Matrix Data:")
        print(csr_matrix_data)
