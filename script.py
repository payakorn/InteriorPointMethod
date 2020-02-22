from main import *
from sparse import create_sparse_matrix, load_data_mps, print_information_sparse, initial_vector_sparse


def test_load_sparse():
    (A, b, c) = ex3()
    # interior(A, b, c, tol=1e-20)
    f, i, j, k, b, n, m = load_data_mps("25FV47.mat")
    print_information_sparse(f, i, j, k, b, n, m)
    x, y, s = initial_vector(A)


def test_create_sparse():
    f, i, j, k, b, n, m = load_data_mps("25FV47.mat")
    print_information_sparse(f, i, j, k, b, n, m)
    x, y, s = initial_vector_sparse(m=m, n=n)
    print('x : ', len(x))
    print('y : ', len(y))
    print('s : ', len(s))
    create_sparse_matrix((i, j, k, m, n), x, s, options='sparse')
    

if __name__ == "__main__":
    test_create_sparse()
