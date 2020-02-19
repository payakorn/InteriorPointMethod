from main import *
from sparse import create_sparse_matrix


(A, b, c) = ex2()
interior(A, b, c, tol=1e-20)
A, b, c = convert_to_array(A, b, c)
x, y, s = initial_vector(A)
sparse_matrix = create_sparse_matrix(A, x, s)
print(sparse_matrix.toarray())
matrix = create_matrix(A, x, y, s)
print(matrix)
print(matrix == sparse_matrix)
i, j = np.shape(sparse_matrix)
print("i :", i, "j :", j)
i, j = np.shape(matrix)
print("i :", i, "j :", j)