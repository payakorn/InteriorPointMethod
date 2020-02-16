import numpy as np
from scipy.optimize import linprog


def generate_system(A, x, s, **options):
    X = np.diagflat(x)
    S = np.diagflat(s)
    block1 = np.block([np.zeros((n, n)), A.T, np.eye(n)])
    block2 = np.block([A, np.zeros((m, m)), np.zeros((m, n))])
    block3 = np.block([S, np.zeros((n, m)), X])

    matrix = np.block([[block1], [block2], [block3]])

    rb = np.matmul(A, x) - b
    rc = np.matmul(A.T, y) + s - c
    if options.get("options") == "predicted":
        r3 = -x * s
        right_hand_side = np.block([[-rc], [-rb], [r3]])
    elif options.get("options") == "corrected":
        delta_x_aff, delta_y_aff, delta_s_aff = direction(A, x, s, options="predicted")
        r4 = (
            (-x * s)
            - (-delta_x_aff * delta_s_aff)
            + (centering * mu_k * np.ones((n, 1)))
        )
        right_hand_side = np.block([[-rc], [-rb], [r4]])
    else:
        raise Exception("No options")
    return (matrix, right_hand_side)


def check_optimality(A, b, c, s):
    primal = e1 * (1 + np.linalg.norm(b)) <= np.linalg.norm(np.matmul(A, x) - b)
    dual = e2 * (1 + np.linalg.norm(c)) <= np.linalg.norm(np.matmul(A.T, y) + s - c)
    duality_gap = e3 <= (np.dot(x.T, s))
    return primal or dual or duality_gap


def direction(A, x, s, options):
    (matrix, right_hand_side) = generate_system(A=A, x=x, s=s, options=options)
    direction = np.linalg.solve(matrix, right_hand_side)
    delta_x_aff = direction[0:n]
    delta_y_aff = direction[n : n + m]
    delta_s_aff = direction[m + n :]
    return (delta_x_aff, delta_y_aff, delta_s_aff)


def convert_to_array(A, b, c):
    A = np.asarray(A)
    b = np.asarray([b]).T
    c = np.asarray([c]).T
    return (A, b, c)


def initial_vector():
    x = np.random.randint(1, high=10, size=(n, 1))
    y = np.random.randint(1, high=10, size=(m, 1))
    s = np.random.randint(1, high=10, size=(n, 1))
    return (x, y, s)


def predicted_stepsize(delta_x_aff, delta_y_aff, delta_s_aff):
    # primal
    i = delta_x_aff < 0
    alpha_primal = min(np.append(-x[i] / delta_x_aff[i], 1))
    # dual
    i = delta_s_aff < 0
    alpha_dual = min(np.append(-s[i] / delta_s_aff[i], 1))
    return (alpha_primal, alpha_dual)


def predicted(x, y, s):
    (alpha_primal, alpha_dual) = predicted_stepsize(
        delta_x_aff=delta_x_aff, delta_y_aff=delta_y_aff, delta_s_aff=delta_s_aff
    )
    x_aff = x + alpha_primal * delta_x_aff
    y_aff = y + alpha_dual * delta_y_aff
    s_aff = s + alpha_dual * delta_s_aff
    return (x_aff, y_aff, s_aff)


def duality_gap():
    (x_aff, y_aff, s_aff) = predicted(x, y, s)
    mu_aff = np.dot(x_aff.T, s_aff) / n
    mu_k = np.dot(x.T, s) / n
    centering = (mu_aff / mu_k) ** 3
    return (mu_aff, mu_k, centering)


def full_stepsize(delta_x, delta_y, delta_s):
    eta = 0.5
    # primal
    i = delta_x_aff < 0
    alpha_primal = min(np.append(-eta * x[i] / delta_x[i], 1))
    # dual
    i = delta_s_aff < 0
    alpha_dual = min(np.append(-eta * s[i] / delta_s[i], 1))
    return (alpha_primal, alpha_dual)

def corrected(x, y, s):
    (alpha_primal, alpha_dual) = full_stepsize(
        delta_x=delta_x, delta_y=delta_y, delta_s=delta_s
    )
    x = x + alpha_primal * delta_x
    y = y + alpha_dual * delta_y
    s = s + alpha_dual * delta_s
    return (x, y, s)


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
scipy_solve(A, b, c)
k = 0
(x, y, s) = initial_vector()
while check_optimality and k<10000:
    (matrix, rb) = generate_system(A, x, s, options="predicted")
    (delta_x_aff, delta_y_aff, delta_s_aff) = direction(A, x, s, options="predicted")
    predicted_stepsize(delta_x_aff, delta_y_aff, delta_s_aff)
    (x_aff, y_aff, s_aff) = predicted(x, y, s)
    (mu_aff, mu_k, centering) = duality_gap()
    (delta_x, delta_y, delta_s) = direction(A, x, s, options="corrected")
    x, y, s = corrected(x, y, s)
    k += 1


print(rb, "\n")
print(matrix, "\n")
print("the optimaility:", check_optimality(A, b, c, s))
print("delta x aff:\n", delta_x_aff)
print("delta y aff:\n", delta_y_aff)
print("delta s aff:\n", delta_s_aff)
print("x_aff:\n", x_aff)
print("y_aff:\n", y_aff)
print("s_aff:\n", s_aff)
print("mu_aff:\n", mu_aff)
print("mu_k:\n", mu_k)
print("centering:\n", centering)
print("delta x:\n", delta_x)
print("delta y:\n", delta_y)
print("delta s:\n", delta_s)
print("x:\n", x)
print("y:\n", y)
print("s:\n", s)
print("k:\n", k)
