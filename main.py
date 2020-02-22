import numpy as np
from scipy.optimize import linprog
from sparse import create_sparse_matrix
from scipy.sparse.linalg import spsolve


def create_matrix(A, x, y, s, **options):
    m, n = np.shape(A)
    X = np.diagflat(x)
    S = np.diagflat(s)
    block1 = np.block([np.zeros((n, n)), A.T, np.eye(n)])
    block2 = np.block([A, np.zeros((m, m)), np.zeros((m, n))])
    block3 = np.block([S, np.zeros((n, m)), X])
    matrix = np.block([[block1], [block2], [block3]])
    return matrix


def create_rhs_predicted(A, b, c, x, y, s, **options):
    rb = np.matmul(A, x) - b
    rc = np.matmul(A.T, y) + s - c
    r3 = -x * s
    right_hand_side = np.block([[-rc], [-rb], [r3]])
    return right_hand_side


def create_rhs_corrected(A, b, c, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff):
    m, n = np.shape(A)
    rb = np.matmul(A, x) - b
    rc = np.matmul(A.T, y) + s - c
    # delta_x_aff, delta_y_aff, delta_s_aff = direction(A, x, y, s, options="predicted")
    mu_aff, mu_k, centering = duality_gap(
        A, x, y, s, delta_x_aff, delta_y_aff, delta_s_aff
    )
    r4 = (-x * s) - (delta_x_aff * delta_s_aff) + (centering * mu_k * np.ones((n, 1)))
    right_hand_side = np.block([[-rc], [-rb], [r4]])
    return right_hand_side


def check_optimality(A, b, c, x, y, s, e1, e2, e3):
    primal = e1 * (1 + np.linalg.norm(b)) <= np.linalg.norm(np.matmul(A, x) - b)
    dual = e2 * (1 + np.linalg.norm(c)) <= np.linalg.norm(np.matmul(A.T, y) + s - c)
    duality_gap_check = e3 >= (np.dot(x.T, s))
    return primal or dual or duality_gap_check


def solve_linear(A, b, method="scipy"):
    if method == "scipy":
        return np.linalg.solve(A, b)
    elif method == "sparse":
        return spsolve(A, b)
    else:
        raise 'no method'


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
    matrix = create_matrix(A=A, x=x, y=y, s=s)
    right_hand_side = create_rhs_predicted(A, b, c, x, y, s)
    direction_vec = solve_linear(matrix, right_hand_side)
    # direction_vec = solve_linear(matrix, right_hand_side, method='sparse')
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
    eta = 0.95
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
    res = linprog(c, A_ub=A, b_ub=b, options={"disp": True},)
    print(res)


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
        print('iteration : {}'.format(k))
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

    # print output 
    print("optimal:", ~check_optimality(A, b, c, x, y, s, e1, e2, e3))
    print("x:\n", x)
    print("k:\n", k)
    print("objective function:", sum(x * c))


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
    c = np.array([-300,-500,-200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    A = np.array([
    [ 10, 7.5,   4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],        #C1
    [  0,  10,   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],        #C2
    [0.5, 0.4, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],        #C3
    [  0, 0.4,   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],        #C4
    [0.5, 0.1, 0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],        #C5
    [0.4, 0.2, 0.4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],        #C6
    [  1, 1.5, 0.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],        #C7
    [  1,   0,   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],        #C8
    [  0,   1,   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],        #C9
    [  0,   0,   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]         #C10
])
    b = np.array([4350, 2500, 280, 140, 280, 140, 700, 300, 180, 400])
    return (A, b, c)
