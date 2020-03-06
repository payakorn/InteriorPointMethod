import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from main import *
from sparse_interior import *


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
    A, b, c = ex3()
    interior(A, b, c)


def create_text_file():
    f = open("guru99.txt", "w+")
    for i in range(10):
        f.write("This is line %d\r\n" % (i + 1))
    f.close()


def test_main_interior_sparse():
    Name, obj_Netlib = benchmark()
    Name_work = benchmark_work()
    # print(Name_work)
    name_benchmark = {}

    # Dict name
    j=0
    for name in Name:
        name_benchmark[name] = obj_Netlib[j]
        j+=1

    line = open("conclusion.txt", "w+")
    line.write(
        "{0:15s} {1:15s} {2:15s} {3:15s} {4:15s} {5:15s}".format(
            "Name", "Obj fun", "Interi time ", "Scipy time", "Interi", "Scipy"
        )
    )
    for i in Name_work:
        print(i)
        A, b, c, cTb = create_problem_from_mps(i)

        # Scipy
        start_time1 = time.time()
        res = scipy_solve(A, b, c)
        end_time1 = time.time()

        # Interior
        start_time2 = time.time()
        # obj_fun = interior_sparse(A=A, b=b, c=c, cTlb=cTb, tol=1e-8)
        obj_fun = new_interior_sparse(Aeq=A, beq=b, c=c, tol=1e-8)
        end_time2 = time.time()

        # information
        print("File name     : {}".format(i))
        print("obj fun Netlib: {0}".format(name_benchmark[i]))
        print("obj fun interi: {0}".format(obj_fun))
        print("obj fun scipy : {0}".format(res.fun))
        print("interior time : {}".format(end_time2 - start_time2))
        print("scipy    time : {}".format(end_time1 - start_time1))
        line.write(
            "{0:15s} {1:15.2f} {2:15.2f} {3:15.2f} {4:15.2f} {5:15.2f}".format(
                i,
                name_benchmark[i],
                end_time2 - start_time2,
                end_time1 - start_time1,
                obj_fun,
                res.fun,
            )
        )
    line.close()


if __name__ == "__main__":
    # test_non_sparse()
    test_main_interior_sparse()
