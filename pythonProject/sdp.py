import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.io import mmread
import gzip
import tarfile
import os


class Node:
    def __init__(self, index):
        self.index = index
        self.result = 0.0
        self.values = []
        self.valueindexes = []
        self.dependencies = []
        self.dependents= []

    def __repr__(self):
        return f"Node(index={self.index}, result={self.result}, dependencies={self.dependencies}, dependents={self.dependents},values={self.values}, valueindexes={self.valueindexes})"

def read_sparse_matrix_from_gz(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    with gzip.open(file_path, 'rt') as f:
        matrix = mmread(f)
        csr = matrix.tocsr()
        return csr

def extract_mtx_files_from_tar_gz(archive_path):
    mtx_files = []
    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.mtx'):
                tar.extract(member)
                mtx_files.append(member.name)
    return mtx_files

def node_analysis(nodes, l):
    e = 1
    levels = []

    processed_nodes = set()
    root_nodes = []
    indexes = []

    # Identify root nodes (nodes with no dependencies)
    for node in nodes:
        if not node.dependencies:
            root_nodes.append(node.index)
            processed_nodes.add(node.index)
            indexes.append(node.index)

    levels.append(root_nodes)

    while len(processed_nodes) < l:
        temp_processed_nodes = []
        temp_indexes = []
        current_level = []

        for node in nodes:
            if node.index not in processed_nodes:
                is_not_dependent = True


                # Check if all dependencies of the current node are in the processed set
                for dep in node.dependencies:
                    if dep not in indexes:
                        is_not_dependent = False
                        break

                if is_not_dependent:
                    current_level.append(node.index)
                    temp_indexes.append(node.index)
                    temp_processed_nodes.append(node.index)


        levels.append(current_level)
        for i in temp_processed_nodes:
            processed_nodes.add(i)
        for i in temp_indexes:
            indexes.append(i)

    return levels

def read_sparse_matrix_csr(file_path):
    with open(file_path, 'r') as file:
        nrows, ncols, nnz = None, None, None
        for line in file:
            if line.startswith('%'):
                continue
            else:
                parts = line.strip().split()
                nrows = int(parts[0])
                ncols = int(parts[1])
                nnz = int(float(parts[2]))
                break

        if nrows is None or ncols is None or nnz is None:
            raise ValueError("Invalid matrix header")

        rows = []
        cols = []
        data = []

        for line in file:
            if not line.startswith('%'):
                row, col, value = map(float, line.strip().split())
                rows.append(int(row) - 1)
                cols.append(int(col) - 1)
                data.append(value)

        csr_matrix_data = sp.csr_matrix((data, (rows, cols)), shape=(nrows, ncols))
        return csr_matrix_data

def solve_sparse_system_with_lu(csr_matrix, b):
    csc_matrix = csr_matrix.tocsc()  # Explicitly convert to CSC format
    lu = splinalg.splu(csc_matrix)
    x = lu.solve(b)
    return x

def build_nodes(csr_matrix):
    nodes = [Node(i) for i in range(csr_matrix.shape[0])]
    for i in range(csr_matrix.shape[0]):
        for j in range(csr_matrix.indptr[i], csr_matrix.indptr[i + 1]):
            if csr_matrix.indices[j] > i:
                continue
            if csr_matrix.indices[j] != i:
                nodes[i].dependencies.append(csr_matrix.indices[j])
                nodes[csr_matrix.indices[j]].dependents.append(i)

            nodes[i].values.append(csr_matrix.data[j])
            nodes[i].valueindexes.append(csr_matrix.indices[j])
    return nodes

if __name__ == "__main__":
    archive_path = "1138_bus.tar.gz"

    if ".tar." in archive_path:
        mtx_files = extract_mtx_files_from_tar_gz(archive_path)
        for mtx_file in mtx_files:
            print(f"Processing file: {mtx_file}")
            csr_matrix_data = read_sparse_matrix_csr(mtx_file)
          #  print("CSR Matrix Data:")
           # print(csr_matrix_data)
            count = 0
            nodes = build_nodes(csr_matrix_data)
            for node in nodes:
                count +=1
                print(node)

            levels = node_analysis(nodes,count)
            level_number = 0
            for level in levels:
                print("Level {} :".format(level_number), level)
                level_number += 1

            b = np.ones(csr_matrix_data.shape[0])
            x = solve_sparse_system_with_lu(csr_matrix_data, b)
         #   print("Solution x:", x)

    elif ".mtx." in archive_path:
        csr_matrix_data = read_sparse_matrix_from_gz(archive_path)
      #  print("CSR Matrix Data:")
       # print(csr_matrix_data)
        count = 0
        nodes = build_nodes(csr_matrix_data)
        for node in nodes:
            count+=1
            print(node)
        levels = node_analysis(nodes,count)
        level_number = 0
        for level in levels:

            print("Level {} :".format(level_number), level)
            level_number+=1
        b = np.ones(csr_matrix_data.shape[0])
        x = solve_sparse_system_with_lu(csr_matrix_data, b)
       # print("Solution x:", x)

    elif archive_path.endswith(".mtx"):
        csr_matrix_data = read_sparse_matrix_csr(archive_path)
        #print("CSR Matrix Data:")
       # print(csr_matrix_data)
        count = 0
        nodes = build_nodes(csr_matrix_data)
        for node in nodes:
            count+=1
            print(node)
        levels = node_analysis(nodes,count)
        level_number = 0
        for level in levels:
            print("Level {} :".format(level_number), level)
            level_number += 1
        b = np.ones(csr_matrix_data.shape[0])
        x = solve_sparse_system_with_lu(csr_matrix_data, b)
        #print("Solution x:", x)
