import numpy as np
from scipy import sparse
from scipy.io import loadmat

# from main import *


def rowcol_to_sparse(i, j, k, m, n):
    return sparse.coo_matrix((k, (i, j)), shape=(m, n))


def create_sparse_matrix(A, x, s, options="non-sparse"):

    """Create sparse matrix in form [0 A^T I]
                                    [A  0  0]
                                    [S  0  X] 
    Arguments:
        A {[mxn array]} -- [sparse : input (i, j, k, m, n)]
        x {[1d array]} -- [solution vector in primal problem]
        s {[1d array]} -- [slack variable in dual problem]
    
    Returns:
        [row index] -- sparse matrix
        [column index] --
        [values] --
    """

    if options == "non-sparse":
        # print("*********create sparse matrix (non-sparse)*********")
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
        col_index = np.append(col_index, range(0, n))
        values = np.append(values, s)
        # X
        row_index = np.append(row_index, range(m + n, m + 2 * n))
        col_index = np.append(col_index, range(m + n, m + 2 * n))
        values = np.append(values, x)
        # check
        # print("sparse matrix non-zero element :")
        # print("row    :", len(row_index))
        # print("col    :", len(col_index))
        # print("values :", len(values))
        return sparse.coo_matrix(
            (values, (row_index, col_index)), shape=(m + 2 * n, m + 2 * n)
        )
        # return sparse.coo_matrix((values, (row_index, col_index)))
    elif options == "sparse":
        # print("***create sparse matrix (sparse)***")
        # try:
        #     i, j, k, m, n = A
        # except:
        i, j, k = sparse.find(A)
        m, n = np.shape(A)
        # print("row              :", len(i))
        # print("col              :", len(j))
        # print("values           :", len(k))
        # print("variables        :", n)
        # print("constraints      :", m)
        # print("number of row    :", max(i))
        # print("number of column :", max(j))
        # A transpose and I
        row_index = np.append(j, range(0, n))
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
        row_index = np.append(row_index, range(m + n, m + 2 * n))
        col_index = np.append(col_index, range(m + n, m + 2 * n))
        values = np.append(values, x)
        # print("****full matrix version****")
        # print("variables           :", m + 2 * n)
        # print("constraints         :", m + 2 * n)
        # print("min index of row    :", min(row_index))
        # print("max index of row    :", max(row_index))
        # print("min index of column :", min(col_index))
        # print("max index of column :", max(col_index))
        return sparse.csc_matrix(
            (values, (row_index, col_index)), shape=(m + 2 * n, m + 2 * n)
        )
        # return sparse.csc_matrix((values, (row_index, col_index)))
    elif options == "tosparse":
        row_index, col_index, values, m, n = A
        return sparse.csc_matrix((values, (row_index, col_index)), shape=(m, n))
    else:
        raise Exception("options must be specific as sparse or non-sparse")


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

    # print("*******load data from mps********")
    data = loadmat("benchmarks/{}".format(file_name))
    return (
        data["f"],
        data["A"]["i"][0][0][0],
        data["A"]["j"][0][0][0],
        data["A"]["k"][0][0][0],
        data["b"],
        data["num_variables"][0][0],
        data["num_constraints"][0][0],
        data["cTlb"][0][0],
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
    print("*********print information**********")
    print("objective function f  :", len(f))
    print("row index             :", len(i))
    print("column index          :", len(j))
    print("values                :", len(k))
    print("right hand side       :", len(b))
    print("number of variables   :", n)
    print("number of condtraints :", m)


def initial_vector_sparse(m, n):
    x = np.ones((n, 1))
    s = np.ones((n, 1))
    # x = np.random.rand(n, 1)
    # x = np.random.randint(low=1, high=n, size=(n, 1))
    # s = np.random.randint(low=1, high=n, size=(n, 1))
    # print("sparse initial\n", x)
    return x, np.ones((m, 1)), s


def resize_parameter(vector):
    row, col = np.shape(vector)
    n = max(row, col)
    if row != n or col != 1:
        vector = np.reshape(vector, (n, -1))
    return vector


def create_problem_from_mps(name):
    c, i, j, k, b, n, m, cTb = load_data_mps("{}.mat".format(name))
    c = resize_parameter(c)
    b = resize_parameter(b)
    A = sparse.csc_matrix((k, (i, j)))
    return A, b, c, cTb


def load_data_mps_full(file_name):

    # print("*******load data from mps********")
    data = loadmat("benchmarks_full/{}".format(file_name))
    c = data["c"]
    try:
        Aineq_i = data["Aineq"]["i"][0][0][0]
        Aineq_j = data["Aineq"]["j"][0][0][0]
        Aineq_k = data["Aineq"]["k"][0][0][0]
        if len(Aineq_i) == 0:
            Aineq_i = None
            Aineq_j = None
            Aineq_k = None
            bineq = None
        else:
            bineq = data["bineq"]
    except:
        Aineq_i = None
        Aineq_j = None
        Aineq_k = None
        bineq = None
    try:
        Aeq_i = data["Aeq"]["i"][0][0][0]
        Aeq_j = data["Aeq"]["j"][0][0][0]
        Aeq_k = data["Aeq"]["k"][0][0][0]
        if len(Aeq_i) == 0:
            Aeq_i = None
            Aeq_j = None
            Aeq_k = None
            beq = None
        else:
            beq = data["beq"]
    except:
        Aeq_i = None
        Aeq_j = None
        Aeq_k = None
        beq = None
    lb = data["lb"]
    ub = data["ub"]
    return c, Aineq_i, Aineq_j, Aineq_k, bineq, Aeq_i, Aeq_j, Aeq_k, beq, lb, ub


def create_problem_from_mps_full(name):
    (
        c,
        Aineq_i,
        Aineq_j,
        Aineq_k,
        bineq,
        Aeq_i,
        Aeq_j,
        Aeq_k,
        beq,
        lb,
        ub,
    ) = load_data_mps_full("{}.mat".format(name))
    c = resize_parameter(c)
    lb = resize_parameter(lb)
    ub = resize_parameter(ub)
    n = len(c)
    if Aineq_i is not None:
        m_ineq = len(bineq)
        Aineq = sparse.csc_matrix((Aineq_k, (Aineq_i, Aineq_j)), shape=((m_ineq, n)))
        bineq = resize_parameter(bineq)
    else:
        Aineq = None
    if Aeq_i is not None:
        m_eq = len(beq)
        Aeq = sparse.csc_matrix((Aeq_k, (Aeq_i, Aeq_j)), shape=((m_eq, n)))
        beq = resize_parameter(beq)
    else:
        Aeq = None
    return c, Aineq, bineq, Aeq, beq, lb, ub


def create_problem_from_mps_matlab(name):
    c, Aineq, bineq, Aeq, beq, lb, ub = load_data_mps_matlab("{}.mat".format(name))
    if len(bineq) == 0:
        Aineq = None
        bineq = None
    if len(beq) == 0:
        Aeq = None
        beq = None
    return c, Aineq, bineq, Aeq, beq, lb, ub


def load_data_mps_matlab(file_name):
    data = loadmat("benchmarks_full/{}".format(file_name))
    Aineq = data["data"]["Aineq"][0][0]
    Aeq = data["data"]["Aeq"][0][0]
    bineq = data["data"]["bineq"][0][0]
    beq = data["data"]["beq"][0][0]
    lb = data["data"]["lb"][0][0]
    ub = data["data"]["ub"][0][0]
    c = data["data"]["f"][0][0]
    return c, Aineq, bineq, Aeq, beq, lb, ub


def csr_vappend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a

def csr_happend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a