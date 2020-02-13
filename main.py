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
    return (matrix, rb)


def check_optimality(A, b, c, s):
    primal = e1 * (1 + np.linalg.norm(b)) <= np.linalg.norm(np.matmul(A, x) - b)
    dual = e2 * (1 + np.linalg.norm(c)) <= np.linalg.norm(np.matmul(A.T, y) + s - c)
    duality_gap = e3 <= (np.dot(x.T, s))
    return primal or dual or duality_gap


# def predicted_newton_direction(matrix, )

## Define problem
c = [-100, -125]
A = [[3, 6], [8, 4], [10, 1]]
b = [30, 44, 55]
# c = np.matrix([-100, -125])
# A = np.matrix([(3, 6), (8, 4)])
# b = np.matrix([30, 44])

## convert to numpy array
A = np.asarray(A)
b = np.asarray([b]).T
c = np.asarray([c]).T

## variables bound
# x0_bounds = (0, 5)
# x1_bounds = (0, 4)
x0_bounds = (0, np.inf)
x1_bounds = (0, np.inf)

# res = linprog(c, A_ub=A, b_ub=b, \
#               bounds=(x0_bounds, x1_bounds),
#               options={"disp": True})

# print(res)

# parameters
n = np.size(A, 1)
m = np.size(A, 0)
tau = 100
rho = 1 / 2
e1 = 0.1
e2 = 0.1
e3 = 0.1

# main

# initial vectors
x = np.ones((n, 1))
y = np.ones((m, 1))
s = np.ones((n, 1))
k = 0

print(check_optimality(A, b, c, s))

(matrix, rb) = generate_system(A, x, s)
print(rb.T, "\n")
print(matrix, "\n")
