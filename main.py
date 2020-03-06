import numpy as np
from scipy.optimize import linprog
from sparse_interior import (
    create_sparse_matrix,
    rowcol_to_sparse,
    initial_vector_sparse,
)
from scipy.sparse.linalg import spsolve
from scipy import sparse


def create_matrix(A, x, y, s, **options):
    m, n = np.shape(A)
    X = np.diagflat(x)
    S = np.diagflat(s)
    block1 = np.block([np.zeros((n, n)), A.T, np.eye(n)])
    block2 = np.block([A, np.zeros((m, m)), np.zeros((m, n))])
    block3 = np.block([S, np.zeros((n, m)), X])
    matrix = np.block([[block1], [block2], [block3]])
    return matrix


def test_create_rhs_predicted(A, b, c, x, y, s, options="non-sparse"):
    # print("shape A :", np.shape(A))
    # print("shape b", np.shape(b))
    # print("shape x", np.shape(x))
    # print("shape y", np.shape(y))
    # print("shape s", np.shape(s))
    # c = np.linalg.transpose(c)
    i, j, k, m, n = A
    A = rowcol_to_sparse(i, j, k, m, n)
    rb = A @ x
    # print(rb)
    # print("shape c :", np.shape(c))
    # print("shape s :", np.shape(s))
    # print("shape y :", np.shape(y))
    # print("shape sparse.coo_matrix.transpose(A) @ y :", np.shape(sparse.coo_matrix.transpose(A) @ y))
    # print(c)
    rowlen_c, collen_c = np.shape(c)
    # print(np.add(sparse.coo_matrix.transpose(A) @ y + s, c.T))
    rc = sparse.coo_matrix.transpose(A) @ y + s - c.T
    # rc = sparse.csc_matrix.transpose(A) @ y
    # print("print : rc = \n")
    # print(rc)
    r3 = x * s
    # print(r3)
    if options == "seperated":
        return rc, rb, r3
    elif options == "full":
        right_hand_side = np.block([[-rc], [-rb], [-r3]])
        return right_hand_side


def create_rhs_predicted(A, b, c, x, y, s, options="non-sparse"):
    # print("shape A :", np.shape(A))
    # print("shape b", np.shape(b))
    r3 = x * s
    if options == "non-sparse":
        try:
            rb = np.matmul(A, x) - b
            rc = np.matmul(A.T, y) + s - c
        except:
            rb = A @ x
            rc = A.T @ y + s - c
        r3 = x * s
        right_hand_side = np.block([[-rc], [-rb], [-r3]])
        return right_hand_side
    elif options == "seperated":
        i, j, k, m, n = A
        A = rowcol_to_sparse(i, j, k, m, n)
        rb = A @ x - b
        rc = sparse.coo_matrix.transpose(A) @ y + s - c
        r3 = x * s
        return rc, rb, r3
    elif options == "full":
        i, j, k, m, n = A
        A = rowcol_to_sparse(i, j, k, m, n)
        rb = A @ x - b
        r3 = x * s
        row_Ax, col_Ax = np.shape(A @ x)
        rc = sparse.coo_matrix.transpose(A) @ y + s - c
        right_hand_side = np.block([[-rc], [-rb], [-r3]])
        return right_hand_side


def create_rhs_corrected(
    A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff, options="non-sparse"
):
    if options == "non-sparse":
        m, n = np.shape(A)
        rb = np.matmul(A, x) - b
        rc = np.matmul(A.T, y) + s - c
        # delta_x_aff, delta_y_aff, delta_s_aff = direction(A, x, y, s, options="predicted")
        mu_aff, mu_k, centering = duality_gap(
            A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        r4 = (
            (-x * s)
            - (delta_x_aff * delta_s_aff)
            + (centering * mu_k * np.ones((n, 1)))
        )
        right_hand_side = np.block([[-rc], [-rb], [r4]])
        return right_hand_side
    elif options == "seperated":
        i, j, k, m, n = A
        A = rowcol_to_sparse(i, j, k, m, n)
        rb = A @ x - b
        rc = sparse.coo_matrix.transpose(A) @ y + s - c.T
        r4 = (
            (x * s) + (delta_x_aff * delta_s_aff) - (centering * mu_k * np.ones((n, 1)))
        )
        mu_aff, mu_k, centering = duality_gap(
            A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        return rc, rb, r4
    elif options == "full":
        i, j, k, m, n = A
        A = rowcol_to_sparse(i, j, k, m, n)
        rb = A @ x - b
        rc = sparse.coo_matrix.transpose(A) @ y + s - c
        mu_aff, mu_k, centering = duality_gap(
            A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        r4 = (
            (x * s) + (delta_x_aff * delta_s_aff) - (centering * mu_k * np.ones((n, 1)))
        )
        row, col = np.shape(delta_x_aff)
        right_hand_side = np.block([[-rc], [-rb], [-r4]])
        return right_hand_side


def check_optimality(A, b, c, x, y, s, e1, e2, e3, options="non-sparse"):
    if options == "non-sparse":
        primal = e1 * (1 + np.linalg.norm(b)) <= np.linalg.norm(np.matmul(A, x) - b)
        dual = e2 * (1 + np.linalg.norm(c)) <= np.linalg.norm(np.matmul(A.T, y) + s - c)
        duality_gap_check = e3 >= (np.dot(x.T, s))
        return primal or dual or duality_gap_check
    elif options == "sparse":
        primal = e1 * (1 + np.linalg.norm(b)) <= np.linalg.norm(A @ x - b)
        dual = e2 * (1 + np.linalg.norm(c)) <= np.linalg.norm(A.T @ y + s - c)
        duality_gap_check = e3 >= (np.dot(x.T, s))
        return primal or dual or duality_gap_check


def solve_linear(A, b, method="scipy"):
    if method == "scipy":
        return np.linalg.solve(A, b)
    elif method == "sparse":
        return spsolve(A, b).reshape((len(b), 1))
    else:
        raise "no method"


def direction_predicted(A, b, c, x, y, s):
    m, n = np.shape(A)
    matrix = create_matrix(A=A, x=x, y=y, s=s)
    right_hand_side = create_rhs_predicted(A, b, c, x, y, s)
    direction_vec = solve_linear(matrix, right_hand_side)
    # direction_vec = solve_linear(matrix, right_hand_side, method='sparse')
    delta_x_aff = direction_vec[0:n]
    delta_y_aff = direction_vec[n : n + m]
    delta_s_aff = direction_vec[m + n :]
    return (delta_x_aff, delta_y_aff, delta_s_aff)


def direction_predicted_sparse(A, b, c, x, y, s):
    m, n = np.shape(A)
    i, j, k = sparse.find(A)
    matrix = create_sparse_matrix(A, x, s, options="sparse")
    right_hand_side = create_rhs_predicted(
        (i, j, k, m, n), b, c, x, y, s, options="full"
    )
    direction_vec = solve_linear(matrix, right_hand_side, method="sparse")
    delta_x_aff = direction_vec[0:n]
    delta_y_aff = direction_vec[n : n + m]
    delta_s_aff = direction_vec[m + n :]
    return (delta_x_aff, delta_y_aff, delta_s_aff)


def direction_corrected(A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff):
    m, n = np.shape(A)
    matrix = create_matrix(A, x, y, s)
    # matrix = create_sparse_matrix(A, x, s)
    right_hand_side = create_rhs_corrected(
        A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
    )
    direction_vec = solve_linear(matrix, right_hand_side)
    # direction_vec = solve_linear(matrix, right_hand_side, method='sparse')
    delta_x = direction_vec[0:n]
    delta_y = direction_vec[n : n + m]
    delta_s = direction_vec[m + n :]
    return delta_x, delta_y, delta_s


def direction_corrected_sparse(A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff):
    m, n = np.shape(A)
    i, j, k = sparse.find(A)
    matrix = create_sparse_matrix(A, x, s, options="sparse")
    right_hand_side = create_rhs_corrected(
        (i, j, k, m, n),
        b,
        c,
        x,
        y,
        s,
        delta_x_aff,
        delta_y_aff,
        delta_s_aff,
        options="full",
    )
    direction_vec = solve_linear(matrix, right_hand_side, method="sparse")
    delta_x_aff = direction_vec[0:n]
    delta_y_aff = direction_vec[n : n + m]
    delta_s_aff = direction_vec[m + n :]
    return (delta_x_aff, delta_y_aff, delta_s_aff)


def convert_to_array(A, b, c):
    A = np.asarray(A)
    b = np.asarray([b]).T
    c = np.asarray([c]).T
    return (A, b, c)


def initial_vector(A):
    m, n = np.shape(A)
    # x = np.zeros((n, 1))
    y = np.zeros((m, 1))
    # I = c < 0
    # x[I] = np.ones(sum(I)[0])
    s = np.ones((n, 1))
    # I = b > 0
    # y[I] = np.ones(len(I))
    # x = np.random.randint(1, high=10, size=(n, 1))
    # y = np.random.randint(1, high=10, size=(m, 1))
    # s = np.random.randint(1, high=10, size=(n, 1))
    x = np.ones((n, 1))
    # y = np.ones((m, 1))
    # s = np.ones((n, 1))
    return (x, y, s)


def predicted_stepsize(delta_x_aff, delta_y_aff, delta_s_aff, x, s):
    # primal
    i = delta_x_aff < 0
    alpha_primal = min(np.append(-x[i] / delta_x_aff[i], 1)) if any(i) else 1
    # dual
    i = delta_s_aff < 0
    alpha_dual = min(np.append(-s[i] / delta_s_aff[i], 1)) if any(i) else 1
    return (alpha_primal, alpha_dual)


def predicted(x, y, s, delta_x_aff, delta_y_aff, delta_s_aff):
    (alpha_primal, alpha_dual) = predicted_stepsize(
        delta_x_aff=delta_x_aff,
        delta_y_aff=delta_y_aff,
        delta_s_aff=delta_s_aff,
        x=x,
        s=s,
    )
    x_aff = x + alpha_primal * delta_x_aff
    y_aff = y + alpha_dual * delta_y_aff
    s_aff = s + alpha_dual * delta_s_aff
    return (x_aff, y_aff, s_aff)


def duality_gap(A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff):
    m, n = np.shape(A)
    (x_aff, y_aff, s_aff) = predicted(x, y, s, delta_x_aff, delta_y_aff, delta_s_aff)
    mu_aff = np.dot(x_aff.T, s_aff) / n
    mu_k = np.dot(x.T, s) / n
    centering = (mu_aff / mu_k) ** 3
    return (mu_aff, mu_k, centering)


def full_stepsize(
    x, y, s, delta_x, delta_y, delta_s, delta_x_aff, delta_y_aff, delta_s_aff
):
    eta = 0.91
    # primal
    i = delta_x < 0
    alpha_primal_max = min(np.append(-x[i] / delta_x[i], 1))
    alpha_primal = min(1, eta * alpha_primal_max)
    # dual
    i = delta_s < 0
    alpha_dual_aff = min(np.append(-s[i] / delta_s[i], 1))
    alpha_dual = min(1, eta * alpha_dual_aff)
    return (alpha_primal, alpha_dual)


def corrected(
    x, y, s, delta_x, delta_y, delta_s, delta_x_aff, delta_y_aff, delta_s_aff
):
    (alpha_primal, alpha_dual) = full_stepsize(
        x, y, s, delta_x, delta_y, delta_s, delta_x_aff, delta_y_aff, delta_s_aff
    )
    x = x + alpha_primal * delta_x
    y = y + alpha_dual * delta_y
    s = s + alpha_dual * delta_s
    return (x, y, s)


def scipy_solve(A, b, c):
    x0_bounds = (0, np.inf)
    x1_bounds = (0, np.inf)
    res = linprog(c, A_eq=A, b_eq=b, method="interior-point", options={"disp": False},)
    return res


def interior(A, b, c, tol=1e-20):
    """[summary]
    
    Arguments:
        A {[array matrix]} -- [matrix constraint]
        b {[array]} -- [right hand side]
        c {[array]} -- [objective function]
    
    Keyword Arguments:
        tol {[float]} -- [error] (default: {1e-20})
    """
    (A, b, c) = convert_to_array(A, b, c)
    e1 = tol
    e2 = tol
    e3 = tol
    m, n = np.shape(A)
    k = 0
    (x, y, s) = initial_vector(A)
    while check_optimality(A, b, c, x, y, s, e1, e2, e3) and k < 50000:
        print("iteration : {}".format(k))
        # get direction
        (delta_x_aff, delta_y_aff, delta_s_aff) = direction_predicted(A, b, c, x, y, s)
        # get stepsize
        (alpha_primal, alpha_dual) = predicted_stepsize(
            delta_x_aff, delta_y_aff, delta_s_aff, x, s
        )
        # update direction
        (x_aff, y_aff, s_aff) = predicted(
            x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # calculate duality gap
        (mu_aff, mu_k, centering) = duality_gap(
            A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # get direction
        (delta_x, delta_y, delta_s) = direction_corrected(
            A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # update direction
        (x, y, s) = corrected(
            x, y, s, delta_x, delta_y, delta_s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # go to the next iteration
        k += 1
        print("objective function:", sum(x * c))

    # print output
    print("optimal:", ~check_optimality(A, b, c, x, y, s, e1, e2, e3))
    print("x:\n", x)
    print("k:\n", k)
    print("objective function:", sum(x * c))


def interior_sparse(A, b, c, cTlb, tol=1e-20):
    """[summary]
    
    Arguments:
        A {[array matrix]} -- [matrix constraint]
        b {[array]} -- [right hand side]
        c {[array]} -- [objective function]
    
    Keyword Arguments:
        tol {[float]} -- [error] (default: {1e-20})
    """
    # (A, b, c) = convert_to_array(A, b, c)
    e1 = tol
    e2 = tol
    e3 = tol
    m, n = np.shape(A)
    k = 0
    # (x, y, s) = initial_vector(A)
    (x, y, s) = initial_vector_sparse(m, n)
    print("solving...")
    while check_optimality(A, b, c, x, y, s, e1, e2, e3, options="sparse") and k < 5000:
        # print("iteration : {}".format(k))
        # get direction
        (delta_x_aff, delta_y_aff, delta_s_aff) = direction_predicted_sparse(
            A, b, c, x, y, s
        )
        # get stepsize
        (alpha_primal, alpha_dual) = predicted_stepsize(
            delta_x_aff, delta_y_aff, delta_s_aff, x, s
        )
        # update direction
        (x_aff, y_aff, s_aff) = predicted(
            x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # calculate duality gap
        (mu_aff, mu_k, centering) = duality_gap(
            A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # get direction
        (delta_x, delta_y, delta_s) = direction_corrected_sparse(
            A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # update direction
        (x, y, s) = corrected(
            x, y, s, delta_x, delta_y, delta_s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # go to the next iteration
        k += 1
        if np.mod(k, 100) == 0:
            print("objective function:", sum(x * c))

    # print output
    print("optimal:", ~check_optimality(A, b, c, x, y, s, e1, e2, e3, options="sparse"))
    # print("x:\n", x)
    print("k:\n", k)
    return (sum(x * c) - cTlb)[0]


def get_Abc(c, Aeq=None, beq=None, Aineq=None, bineq=None):
    if Aeq is not None and Aineq is not None:
        row_Aeq, col_Aeq = np.shape(Aeq)
        row_Aineq, col_Aineq = np.shape(Aineq)
        if not sparse.issparse(Aineq):
            I = np.eye(row_Aineq)
            block1 = np.block([Aineq, I])
            block2 = np.block([Aeq, np.zeros((row_Aeq, row_Aineq))])
            A = np.block([[block1], [block2]])
        else:  # sparse
            row_Aineq, col_Aineq, value_Aineq = sparse.find(Aineq)
            row_Aeq, col_Aeq, value_Aeq = sparse.find(Aeq)
            # create slack I
            row_I = range(row_Aineq)
            col_I = range(col_Aineq, col_Aineq + row_Aineq)
            value_I = np.ones((row_Aineq, 1))
            # create Aeq
            row_Aeq += row_Aineq
            # append all
            row = np.block([row_Aineq, row_Aeq, row_I])
            col = np.block([col_Aineq, col_Aeq, col_I])
            value = np.block([value_Aineq, value_Aeq, value_I])
            A = sparse.csc_matrix((value, (row, col)))
        b = np.block([[bineq], [beq]])
        return A, b
    elif Aeq is not None:
        return Aeq, beq
    elif Aineq is not None:
        row_Aineq, col_Aineq = np.shape(Aineq)
        if not sparse.issparse(Aineq):
            I = np.eye(row_Aineq)
            block1 = np.block([Aineq, I])
        else:
            I = sparse.identity(row_Aineq)
            A = sparse.hstack([Aineq, I])
            # row_Aineq, col_Aineq, value_Aineq = sparse.find(Aineq)
            # #  Create I
            # row_I = range(row_Aineq)
            # col_I = range(col_Aineq, col_Aineq + row_Aineq)
            # value_I = np.ones((row_Aineq, 1))
            return A, bineq


def new_interior_sparse(
    c, Aeq=None, beq=None, Aineq=None, bineq=None, lb=None, ub=None, tol=1e-20
):
    e1 = tol
    e2 = tol
    e3 = tol
    A, b = get_Abc(c, Aeq, beq, Aineq, bineq)
    m, n = np.shape(A)
    k = 0
    # (x, y, s) = initial_vector(A)
    (x, y, s) = initial_vector_sparse(m, n)
    print("solving...")
    while check_optimality(A, b, c, x, y, s, e1, e2, e3, options="sparse") and k < 5000:
        # print("iteration : {}".format(k))
        # get direction
        (delta_x_aff, delta_y_aff, delta_s_aff) = direction_predicted_sparse(
            A, b, c, x, y, s
        )
        # get stepsize
        (alpha_primal, alpha_dual) = predicted_stepsize(
            delta_x_aff, delta_y_aff, delta_s_aff, x, s
        )
        # update direction
        (x_aff, y_aff, s_aff) = predicted(
            x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # calculate duality gap
        (mu_aff, mu_k, centering) = duality_gap(
            A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # get direction
        (delta_x, delta_y, delta_s) = direction_corrected_sparse(
            A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # update direction
        (x, y, s) = corrected(
            x, y, s, delta_x, delta_y, delta_s, delta_x_aff, delta_y_aff, delta_s_aff
        )
        # go to the next iteration
        k += 1
        if np.mod(k, 100) == 0:
            print("objective function:", sum(x * c))

    # print output
    print("optimal:", ~check_optimality(A, b, c, x, y, s, e1, e2, e3, options="sparse"))
    # print("x:\n", x)
    print("k:\n", k)
    return (sum(x * c))[0]


## Example problems
def ex1():
    c = [-100, -125, -20]
    A = [[3, 6, 8], [8, 4, 1]]
    b = [30, 44]
    # solution f = -775
    return (A, b, c)


def ex2():
    c = [-20, -30, 0, 0, 0]
    A = [[1, 1.5, 1, 0, 0], [2, 3, 0, 1, 0], [2, 1, 0, 0, 1]]
    b = [750, 1500, 1000]
    # solution f = -15000, x_star = (300, 300, ...)
    return (A, b, c)


def ex3():
    c = np.array([-300, -500, -200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    A = np.array(
        [
            [10, 7.5, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C1
            [0, 10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # C2
            [0.5, 0.4, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # C3
            [0, 0.4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # C4
            [0.5, 0.1, 0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # C5
            [0.4, 0.2, 0.4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # C6
            [1, 1.5, 0.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # C7
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # C8
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # C9
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # C10
        ]
    )
    b = np.array([4350, 2500, 280, 140, 280, 140, 700, 300, 180, 400])
    return (A, b, c)


def benchmark():
    abc = [
        "25FV47",
        "80BAU3B",
        "ADLITTLE",
        "AFIRO",
        "AGG",
        "AGG2",
        "AGG3",
        "BANDM",
        "BEACONFD" "BLEND",
        "BNL1",
        "BNL2",
        "BOEING1",
        "BOEING2",
        "BORE3D",
        "BRANDY",
        "CAPRI",
        "CYCLE",
        "CZPROB",
        "D2Q06C",
        "D6CUBE",
        "DEGEN2",
        "DEGEN3",
        "DFL001",
        "E226",
        "ETAMACRO",
        "FFFFF800",
        "FINNIS",
        "FIT1D",
        "FIT1P",
        "FIT2D",
        "FIT2P",
        "FORPLAN",
        "GANGES",
        "GFRD-PNC",
        "GREENBEA",
        "GREENBEB",
        "GROW15",
        "GROW22",
        "GROW7",
        "ISRAEL",
        "KB2",
        "LOTFI",
        "MAROS",
        "MAROS-R7",
        "MODSZK1",
        "NESM",
        "PEROLD",
        "PILOT",
        "PILOT.JA",
        "PILOT.WE",
        "PILOT4",
        "PILOT87",
        "PILOTNOV",
        "QAP8",
        "QAP12",
        "QAP15",
        "RECIPE",
        "SC105",
        "SC205",
        "SC50A",
        "SC50B",
        "SCAGR25",
        "SCAGR7",
        "SCFXM1",
        "SCFXM2",
        "SCFXM3",
        "SCORPION",
        "SCRS8",
        "SCSD1",
        "SCSD6",
        "SCSD8",
        "SCTAP1",
        "SCTAP2",
        "SCTAP3",
        "SEBA",
        "SHARE1B",
        "SHARE2B",
        "SHELL",
        "SHIP04L",
        "SHIP04S",
        "SHIP08L",
        "SHIP08S",
        "SHIP12L",
        "SHIP12S",
        "SIERRA",
        "STAIR",
        "STANDATA",
        "STANDGUB",
        "STANDMPS",
        "STOCFOR1",
        "STOCFOR2",
        "STOCFOR3",
        "TRUSS",
        "TUFF",
        "VTP.BASE",
        "WOOD1P",
        "WOODW",
    ]
    obj_values = [
        5.5018458883e03,
        9.8723216072e05,
        2.2549496316e05,
        -4.6475314286e02,
        -3.5991767287e07,
        -2.0239252356e07,
        1.0312115935e07,
        -1.5862801845e02,
        3.3592485807e04,
        -3.0812149846e01,
        1.9776292856e03,
        1.8112365404e03,
        -3.3521356751e02,
        -3.1501872802e02,
        1.3730803942e03,
        1.5185098965e03,
        2.6900129138e03,
        -5.2263930249e00,
        2.1851966989e06,
        1.2278423615e05,
        3.1549166667e02,
        -1.4351780000e03,
        -9.8729400000e02,
        1.12664e07,
        -1.8751929066e01,
        -7.5571521774e02,
        5.5567961165e05,
        1.7279096547e05,
        -9.1463780924e03,
        9.1463780924e03,
        -6.8464293294e04,
        6.8464293232e04,
        -6.6421873953e02,
        -1.0958636356e05,
        6.9022359995e06,
        -7.2462405908e07,
        -4.3021476065e06,
        -1.0687094129e08,
        -1.6083433648e08,
        -4.7787811815e07,
        -8.9664482186e05,
        -1.7499001299e03,
        -2.5264706062e01,
        -5.8063743701e04,
        1.4971851665e06,
        3.2061972906e02,
        1.4076073035e07,
        -9.3807580773e03,
        -5.5740430007e02,
        -6.1131344111e03,
        -2.7201027439e06,
        -2.5811392641e03,
        3.0171072827e02,
        -4.4972761882e03,
        2.0350000000e02,
        5.2289435056e02,
        1.0409940410e03,
        -2.6661600000e02,
        -5.2202061212e01,
        -5.2202061212e01,
        -6.4575077059e01,
        -7.0000000000e01,
        -1.4753433061e07,
        -2.3313892548e06,
        1.8416759028e04,
        3.6660261565e04,
        5.4901254550e04,
        1.8781248227e03,
        9.0429998619e02,
        8.6666666743e00,
        5.0500000078e01,
        9.0499999993e02,
        1.4122500000e03,
        1.7248071429e03,
        1.4240000000e03,
        1.5711600000e04,
        -7.6589318579e04,
        -4.1573224074e02,
        1.2088253460e09,
        1.7933245380e06,
        1.7987147004e06,
        1.9090552114e06,
        1.9200982105e06,
        1.4701879193e06,
        1.4892361344e06,
        1.5394362184e07,
        -2.5126695119e02,
        1.2576995000e03,
        "(see NOTES)",
        1.4060175000e03,
        -4.1131976219e04,
        -3.9024408538e04,
        -3.9976661576e04,
        4.5881584719e05,
        2.9214776509e-01,
        1.2983146246e05,
        1.4429024116e00,
        1.3044763331e00,
    ]
    name = [
        "25FV47",
        "80BAU3B",
        "ADLITTLE",
        "AFIRO",
        "AGG",
        "AGG2",
        "AGG3",
        "BANDM",
        "BEACONFD",
        "BLEND",
        "BNL1",
        "BNL2",
        "BOEING1",
        "BOEING2",
        "BORE3D",
        "BRANDY",
        "CAPRI",
        "CYCLE",
        "CZPROB",
        "D2Q06C",
        "D6CUBE",
        "DEGEN2",
        "DEGEN3",
        "DFL001",
        "E226",
        "ETAMACRO",
        "FFFFF800",
        "FINNIS",
        "FIT1D",
        "FIT1P",
        "FIT2D",
        "FIT2P",
        "FORPLAN",
        "GANGES",
        "GFRD-PNC",
        "GREENBEA",
        "GREENBEB",
        "GROW15",
        "GROW22",
        "GROW7",
        "ISRAEL",
        "KB2",
        "LOTFI",
        "MAROS",
        "MAROS-R7",
        "MODSZK1",
        "NESM",
        "PEROLD",
        "PILOT",
        "PILOT.JA",
        "PILOT.WE",
        "PILOT4",
        "PILOT87",
        "PILOTNOV",
        "QAP8",
        "QAP12",
        "QAP15",
        "RECIPE",
        "SC105",
        "SC205",
        "SC50A",
        "SC50B",
        "SCAGR25",
        "SCAGR7",
        "SCFXM1",
        "SCFXM2",
        "SCFXM3",
        "SCORPION",
        "SCRS8",
        "SCSD1",
        "SCSD6",
        "SCSD8",
        "SCTAP1",
        "SCTAP2",
        "SCTAP3",
        "SEBA",
        "SHARE1B",
        "SHARE2B",
        "SHELL",
        "SHIP04L",
        "SHIP04S",
        "SHIP08L",
        "SHIP08S",
        "SHIP12L",
        "SHIP12S",
        "SIERRA",
        "STAIR",
        "STANDATA",
        "STANDGUB",
        "STANDMPS",
        "STOCFOR1",
        "STOCFOR2",
        "STOCFOR3",
        "TRUSS",
        "TUFF",
        "VTP.BASE",
        "WOOD1P",
        "WOODW",
    ]
    return name, obj_values


def benchmark_work():
    name = [
        "25FV47",
        "ADLITTLE",
        "AFIRO",
        "AGG2",
        "AGG3",
        "BANDM",
        "BNL1",
        "BNL2",
        "CZPROB",
        "D2Q06C",
        "DEGEN2",
        "DEGEN3",
        "E226",
        "FFFFF800",
        "ISRAEL",
        "LOTFI",
        "MAROS-R7",
        "QAP8",
        "QAP12",
        "QAP15",
        "SC105",
        "SC205",
        "SC50A",
        "SC50B",
        "SCAGR25",
        "SCAGR7",
        "SCFXM1",
        "SCFXM2",
        "SCFXM3",
        "SCORPION",
        "SCRS8",
        "SCSD1",
        "SCSD6",
        "SCSD8",
        "SCTAP1",
        "SCTAP2",
        "SCTAP3",
        "SHARE1B",
        "SHARE2B",
        "STOCFOR1",
        "STOCFOR2",
        "STOCFOR3",
        "TRUSS",
        "WOOD1P",
        "WOODW",
    ]
    return name
