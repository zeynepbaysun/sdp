import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import gzip
import tarfile
import os


def read_sparse_matrix_from_gz(file_name):
    # Get the absolute path to the file in the project directory
    file_path = os.path.join(os.getcwd(), file_name)

    with gzip.open(file_path, 'rt') as f:
        # Read the decompressed file using mmread
        matrix = mmread(f)
        csr = matrix.tocsr()
        return csr


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
        csr_matrix_data = sp.csr_matrix((data, (rows, cols)), shape=(nrows, ncols))
        return csr_matrix_data


def analysis_phase(adjacency_matrix):
    # Initialize the level number
    e = 1

    # Initialize a list to hold nodes for each level
    levels = []

    # Find the total number of nodes
    n = adjacency_matrix.shape[0]

    # Set to keep track of nodes that have been processed
    processed_nodes = set()

    while len(processed_nodes) < n:
        # Find root nodes (nodes with no dependencies)
        root_nodes = []
        for node in range(n):
            if node not in processed_nodes and all(
                    adjacency_matrix[:, node] == 0):
                root_nodes.append(node)

        if not root_nodes:
            print("No root nodes found. There might be a cycle or incorrect dependencies.")
            break

        # Process root nodes
        current_level = []
        for node in root_nodes:
            current_level.append(node)
            processed_nodes.add(node)

        # Add the current level to the list of levels
        levels.append(current_level)

        # Remove the dependencies of the processed nodes
        for node in current_level:
            for i in range(n):
                adjacency_matrix[node, i] = 0

        # Increment the level number
        e += 1

    return levels


# Example usage
if __name__ == "__main__":
    # Path to your .mtx or .mtx.gz file
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

            # Run the analysis phase on the matrix
            adjacency_matrix = csr_matrix_data.toarray()
            levels = analysis_phase(adjacency_matrix)
            print("Levels:", levels)

    elif ".mtx." in archive_path:
        csr_matrix_data = read_sparse_matrix_from_gz(archive_path)
        # Print CSR matrix format
        print("CSR Matrix Data:")
        print(csr_matrix_data)

        # Run the analysis phase on the matrix
        adjacency_matrix = csr_matrix_data.toarray()
        levels = analysis_phase(adjacency_matrix)
        print("Levels:", levels)

    elif archive_path.endswith(".mtx"):
        csr_matrix_data = read_sparse_matrix_csr(archive_path)
        # Print CSR matrix format
        print("CSR Matrix Data:")
        print(csr_matrix_data)

        # Run the analysis phase on the matrix
        adjacency_matrix = csr_matrix_data.toarray()
        levels = analysis_phase(adjacency_matrix)
        print("Levels:", levels)
