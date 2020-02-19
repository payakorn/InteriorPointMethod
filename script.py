from main import *
from sparse import create_sparse_matrix


(A, b, c) = ex2()
interior(A, b, c, tol=1e-20)