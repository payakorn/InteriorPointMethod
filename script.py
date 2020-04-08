import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import spsolve

from main import *
from sparse_interior import *

# import warnings

# warnings.filterwarnings('error')


def test_load_example_sparse():
    A, b, f = ex3()
    f = f.reshape((1, len(f)))
    i, j, k = sparse.find(A)
    m, n = np.shape(A)
    return f, i, j, k, b, n, m


def test_load_sparse(function):
    (A, b, c) = function()
    # interior(A, b, c, tol=1e-20)
    f, i, j, k, b, n, m = load_data_mps("25FV47.mat")
    print_information_sparse(f, i, j, k, b, n, m)
    print(i)
    x, y, s = initial_vector(A)


def test_create_sparse():
    pass
    f, i, j, k, b, n, m = load_data_mps("25FV47.mat")
    print_information_sparse(f, i, j, k, b, n, m)
    x, y, s = initial_vector_sparse(m=m, n=n)
    # print('x : ', len(x))
    # print('y : ', len(y))
    # print('s : ', len(s))
    matrix = create_sparse_matrix((i, j, k, m, n), x, s, options="sparse")
    # print(matrix)
    # print(k)

    # print("shape :", np.shape(matrix))
    # print(matrix @ np.ones((9286)))
    # print(np.matmul(matrix, np.ones(9286)))
    new_matrix = create_sparse_matrix((i, j, k, m, n), x, s, options="tosparse")
    print(new_matrix)
    # b = create_rhs_predicted(new_matrix, b, f, x, y, s)
    # spsolve(new_matrix, b)


def test_spsolve():
    # f, i, j, k, b, n, m = load_data_mps("25FV47.mat")
    # f, i, j, k, b, n, m = load_data_mps("80BAU3B.mat")
    f, i, j, k, b, n, m = load_data_mps("ADLITTLE.mat")
    # f, i, j, k, b, n, m = test_load_example_sparse()
    # print(np.shape(f))
    # print(np.array(f, shape=(n, 1)))
    x, y, s = initial_vector_sparse(m=m, n=n)
    matrix_new = create_sparse_matrix((i, j, k, m, n), x, s, options="sparse")
    # new_matrix = rowcol_to_sparse(i, j, k, m, n)
    # plt.spy(matrix_new, markersize=1)
    # plt.show()
    # print(new_matrix)
    b_new = test_create_rhs_predicted((i, j, k, m, n), b, f, x, y, s, options="full")
    # r1, r2, r3 = test_create_rhs_predicted(new_matrix, b, f, x, y, s, options="seperated")
    spsolve(matrix_new, b_new)
    # plt.spy(new_matrix, markersize=1)
    # plt.show()
    matrix, right_hand_side = create_sparse_eliminate(
        (i, j, k, m, n), b, f, x, y, s, options="sparse"
    )
    print("*********sparse solve**********")
    # print("**information**")
    # size_row, size_col = np.shape(matrix)
    # print("row :", size_row)
    # print("col :", size_col)
    # print("b :", len(right_hand_side))
    solution = spsolve(matrix, right_hand_side)
    print("asdasdasd")
    print("solution :", sum(solution * f))


def test_non_sparse():
    """test run interior point for input non-sparse matrix
    min c^Tx
    s.t. Ax = b
          x >= 0
    """
    A, b, c = ex3()
    interior(A, b, c)


def convert_inf_to_none(x):
    if x[0] == np.inf or x[0] == -np.inf:
        return None
    else:
        return x[0]


def create_bound_for_scipy(lb, ub):
    """scipy require different bound structure i.e. [( (lb_1, ub_1),...,(lb_n, ub_n) )]
    
    Arguments:
        lb {[numpy array]} -- [description]
        ub {[numpy array]} -- [description]
    
    Returns:
        [bound in scipy structure] -- [description]
    """
    lb = tuple(map(convert_inf_to_none, lb))
    ub = tuple(map(convert_inf_to_none, ub))
    return list((lb[i], ub[i]) for i in range(len(ub)))


def test_main_interior_sparse():
    """test interior point method with benchmarks in the form
    min c^Tx
    s.t. Aineq * x <= bineq
          Aeq * x   = beq
          lb <= x <= ub
    """
    bound = None
    Name, obj_Netlib = benchmark()
    Name_work = benchmark_work()
    name_benchmark = {}

    # Dict name
    j = 0
    for name in Name:
        name_benchmark[name] = obj_Netlib[j]
        j += 1

    line = open("conclusion1.txt", "w")
    line.write(
        "{0:17s} {2:>17s} {3:>20s} {1:>20s} {4:>20s} {5:>20s}\r\n".format(
            "Name", "Obj fun", "Interi time", "Scipy time", "Interi", "Scipy"
        )
    )
    line.close()
    for i in Name_work[1:12]:
        print('\n\nProblem name: {}'.format(i))
        c, Aineq, bineq, Aeq, beq, lb, ub = create_problem_from_mps_matlab(i)
        # Scipy
        start_time1 = time.time()
        bounds = create_bound_for_scipy(lb, ub)
        res = linprog(
            c=c,
            A_ub=Aineq,
            b_ub=bineq,
            A_eq=Aeq,
            b_eq=beq,
            bounds=bounds,
            method="interior-point",
            options={"disp": True},
        )
        # res = np.nan
        end_time1 = time.time()

        # Interior
        start_time2 = time.time()
        # obj_fun = interior_sparse(A=A, b=b, c=c, cTlb=cTb, tol=1e-8)
        obj_fun = new_interior_sparse(c=c, Aineq=Aineq, bineq=bineq, Aeq=Aeq, beq=beq, lb=lb, ub=ub, tol=1e-6)
        end_time2 = time.time()

        # information
        print("File name     : {}".format(i))
        print("obj fun Netlib: {0}".format(name_benchmark[i]))
        print("obj fun interi: {0}".format(obj_fun))
        print("obj fun scipy : {0}".format(res.fun))
        # print("obj fun scipy : {0}".format(np.nan))
        print("interior time : {}".format(end_time2 - start_time2))
        print("scipy    time : {}".format(end_time1 - start_time1))
        line = open("conclusion1.txt", "a")
        line.write(
            "{0:17s} {2:17.2f} {3:>20.2f} {1:20.2f} {4:20.2f} {5:20.2f}\r\n".format(
                i,
                name_benchmark[i],
                end_time2 - start_time2,
                end_time1 - start_time1,
                obj_fun,
                res.fun,
                # np.nan
            )
        )
        line.close()


def test_load_matrix():
    c, Aineq, bineq, Aeq, beq, lb, ub = load_data_mps_matlab(file_name='BANDM')
    print(f'{Aineq = }')
    print(f'{sparse.issparse(Aineq) = }')
    print(f'{Aeq = }')
    print(f'{sparse.issparse(Aeq) = }')
    print(f'{len(bineq) = }, shape={np.shape(bineq)}')
    print(f'{len(beq) = }, shape={np.shape(beq)}')
    print(f'{len(lb) = }, shape={np.shape(lb)}')
    print(f'{len(ub) = }, shape={np.shape(ub)}')
    print(f'{len(c) = }, shape={np.shape(c)}')


def check_bound():
    Name, obj_Netlib = benchmark()
    Name_work = benchmark_work()
    for i in Name_work:
        c, Aineq, bineq, Aeq, beq, lb, ub = load_data_mps_matlab(file_name=i)
        ub = np.array([j[0] for j in ub])
        n = len(lb)
        lower_bound_zeros =  np.count_nonzero(np.array([j[0] for j in lb]) == 0)
        upper_bound_inf =  sum(np.isinf(ub))
        print('Name:', i)
        print('lower bound zeros:', lower_bound_zeros)
        print('upper bound inf:', upper_bound_inf)
        if lower_bound_zeros == n:
            print('no lower bound')
        else:
            print('has lower bound')
        if upper_bound_inf == n:
            print('no upper bound\n')
        else:
            print('has upper bound\n')


if __name__ == "__main__":
    test_main_interior_sparse()
    # check_bound()