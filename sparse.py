import numpy as np
from scipy import sparse


def create_sparse_matrix(A, x, s):
    m, n = np.shape(A)
    i, j, k = sparse.find(A)
    # A transpose and I
    row_index = np.append(j, range(m + n, m + 2 * n))
    col_index = np.append(i + n, range(m + n, m + 2 * n))
    values = np.append(k, np.ones(n))
    # A
    row_index = np.append(row_index, i + n)
    col_index = np.append(col_index, j)
    values = np.append(values, k)
    # S
    row_index = np.append(row_index, range(m + n, m + 2 * n))
    col_index = np.append(col_index, range(n))
    values = np.append(values, s)
    # X
    row_index = np.append(row_index, range(n))
    col_index = np.append(col_index, range(m + n, m + 2 * n))
    values = np.append(values, x)
    # check
    print("row :", len(row_index))
    print("col :", len(col_index))
    print("values :", len(values))
    return sparse.coo_matrix(
        (values, (row_index, col_index)), shape=(m + 2 * n, m + 2 * n)
    )


# get value from sparse matrix
def get_item(row_index, column_index, matrix):
    # Get row values
    row_start = matrix.indptr[row_index]
    row_end = matrix.indptr[row_index + 1]
    row_values = matrix.data[row_start:row_end]

    # Get column indices of occupied values
    index_start = matrix.indptr[row_index]
    index_end = matrix.indptr[row_index + 1]

    # contains indices of occupied cells at a specific row
    row_indices = list(matrix.indices[index_start:index_end])

    # Find a positional index for a specific column index
    value_index = row_indices.index(column_index)

    if value_index >= 0:
        return row_values[value_index]
    else:
        # non-zero value is not found
        return 0


