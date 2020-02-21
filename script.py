from main import *
from sparse import create_sparse_matrix, load_data_mps, print_information_sparse


(A, b, c) = ex3()
# interior(A, b, c, tol=1e-20)
f, i, j, k, b, n, m = load_data_mps("obj_fun.mat")
print_information_sparse(f, i, j, k, b, n, m)
x, y, s = initial_vector(A)