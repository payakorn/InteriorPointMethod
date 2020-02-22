import numpy as np
from scipy import sparse
from scipy.io import loadmat


def create_sparse_matrix(A, x, s, options="non-sparse"):

    """Create sparse matrix in form [0 A^T I]
                                    [A  0  0]
                                    [S  0  X] 
    Arguments:
        A {[mxn array]} -- [linear constraint Ax = b]
        x {[1d array]} -- [solution vector in primal problem]
        s {[1d array]} -- [slack variable in dual problem]
    
    Returns:
        [row index] -- sparse matrix
        [column index] --
        [values] --
    """

    if options == "non-sparse":
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
        print("sparse matrix non-zero element :")
        print("row    :", len(row_index))
        print("col    :", len(col_index))
        print("values :", len(values))
        return sparse.coo_matrix(
            (values, (row_index, col_index)), shape=(m + 2 * n, m + 2 * n)
        )
    elif options == 'sparse':
        print("***sparse***")
        i, j, k, m, n = A
        print("row    :", len(i))
        print("col    :", len(j))
        print("values :", len(k))
        print("variables :", n)
        print("constraints :", m)
        print("number of row :", max(i))
        print("number of column :", max(j))
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
        print('****full matrix version****')
        print("variables :", m+2*n)
        print("constraints :", m+2*n)
        print("max index of row :", max(row_index))
        print("max index of column :", max(col_index))       
        return sparse.coo_matrix(
            (values, (row_index, col_index)), shape=(m + 2 * n, m + 2 * n)
        )
    else:
        raise Exception('options must be specific as sparse or non-sparse')


# get value from sparse matrix
def get_item(row_index, column_index, matrix):
    """get value from sparse matrix
    
    Arguments:
        row_index {integer} -- row
        column_index {integer} -- column
        matrix {sparse matrix} -- --
    
    Returns:
        float -- value
    """
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


def load_data_mps(file_name):

    """load linear problem from .mat file (standard form), i.e. min f^Tx s.t. Ax = b
    
    Arguments:
        file_name {text} -- .mat file
    
    Returns:
        f -- objective function
        i {array} -- row index for sparse matrix A
        j {array} -- column index for sparse matrix A
        k {array} -- values for sparse matrix A
        b {array} -- right hand side vector s.t. Ax = b
        n {number} -- the number of variables
        m {number} -- the number of constraints
    """

    data = loadmat(file_name)
    return (
        data["f"],
        data["A"]["i"][0][0],
        data["A"]["j"][0][0],
        data["A"]["k"][0][0],
        data["b"],
        data["num_variables"][0][0],
        data["num_constraints"][0][0],
    )


def print_information_sparse(f, i, j, k, b, n, m):

    """[print the information about linear programming problem]
    
    Arguments:
        f {array} -- objective function
        i {array} -- row index for sparse matrix A
        j {array} -- column index for sparse matrix A
        k {array} -- values for sparse matrix A
        b {array} -- right hand side vector s.t. Ax = b
        n {number} -- the number of variables
        m {number} -- the number of constraints
    """
    print("objective function f  :", len(f))
    print("row index             :", len(i))
    print("column index          :", len(j))
    print("values                :", len(k))
    print("right hand side       :", len(b))
    print("number of variables   :", n)
    print("number of condtraints :", m)


def initial_vector_sparse(m, n):
    return np.ones((n, 1)), np.ones((m, 1)), np.ones((n, 1))
