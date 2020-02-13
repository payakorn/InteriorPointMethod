import numpy as np
from scipy.optimize import linprog


def generate_system(A, x, s):
    X = np.diagflat(x)
    S = np.diagflat(s)
    block1 = np.block([np.zeros((n, n)), A.T, np.eye(n)])
    block2 = np.block([A, np.zeros((m, m)), np.zeros((m, n))])
    block3 = np.block([S, np.zeros((n, m)), X])

    matrix = np.block([[block1], [block2], [block3]])

    rb = np.matmul(A, x) - b
    rc = np.matmul(A.T, y) + s - c
    r3 = np.ones((n, 1))
    r4 = np.matmul(S, X)
    right_hand_side = np.block([[rc], [rb], [r3]])
    return (matrix, right_hand_side)


def check_optimality(A, b, c, s):
    primal = e1 * (1 + np.linalg.norm(b)) <= np.linalg.norm(np.matmul(A, x) - b)
    dual = e2 * (1 + np.linalg.norm(c)) <= np.linalg.norm(np.matmul(A.T, y) + s - c)
    duality_gap = e3 <= (np.dot(x.T, s))
    return primal or dual or duality_gap


def predicted_newton_direction(A, x, s):
    (matrix, right_hand_side) = generate_system(A, x, s)
    print("det =", np.linalg.det(matrix))
    direction = np.linalg.solve(matrix, right_hand_side)
    delta_x = direction[0:n]
    delta_y = direction[n : n + m]
    delta_s = direction[m + n :]
    return (delta_x, delta_y, delta_s)


def convert_to_array(A, b, c):
    A = np.asarray(A)
    b = np.asarray([b]).T
    c = np.asarray([c]).T
    return (A, b, c)


def initial_vector():
    x = np.random.randint(10, size=(n, 1))
    y = np.random.randint(10, size=(m, 1))
    s = np.random.randint(10, size=(n, 1))
    return (x, y, s)


def predicted_stepsize(delta_x, delta_y, delta_s):
    pass


## Define problem
c = [-100, -125, -20]
A = [[3, 6, 8], [8, 4, 1]]
b = [30, 44]

## convert to numpy array
(A, b, c) = convert_to_array(A, b, c)


def scipy_solve(A, b, c):
    x0_bounds = (0, np.inf)
    x1_bounds = (0, np.inf)
    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=(x0_bounds, x1_bounds, x1_bounds),
        options={"disp": True},
    )
    print(res)


# parameters
n = np.size(A, 1)
m = np.size(A, 0)
tau = 100
rho = 1 / 2
e1 = 0.1
e2 = 0.1
e3 = 0.1

# main
k = 0
(x, y, s) = initial_vector()
(matrix, rb) = generate_system(A, x, s)
scipy_solve(A, b, c)
(delta_x, delta_y, delta_s) = predicted_newton_direction(A, x, s)


print(rb, "\n")
print(matrix, "\n")
print("the optimaility:", check_optimality(A, b, c, s))
print("newton direction:\n", direction)
print("delta x:\n", delta_x)
print("delta y:\n", delta_y)
print("delta s:\n", delta_s)
