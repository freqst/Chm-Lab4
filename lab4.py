import numpy as np
np.random.seed(42)#Фиксирование рандомайзера
# Генерация матрицы A0 и вектора b
n = 4 # Размерность матрицы
A0 = np.random.uniform(-1.0, 1.0, (n, n))
A0 = (A0 + A0.T) / 2  # Симметризация
m = 2  # Параметр m
E = np.eye(n)
A = A0 + m * E  # Матрица A
b = np.random.uniform(-1.0, 1.0, n)  # Вектор b
print(A)
# Степенной метод
def power_method(A, epsilon= 1e-6, max_iter=1000):
    n = A.shape[0]
    x = np.ones(n)  # Начальный вектор
    lambda_prev = 0  # Предыдущее собственное значение
    for _ in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        lambda_new = x_new.T @ A @ x_new
        if np.abs(lambda_new - lambda_prev) < epsilon:
            break
        x = x_new
        lambda_prev = lambda_new
    return lambda_new, x_new

lambda_max, eigenvector = power_method(A)
print("Наибольшее собственное значение:", lambda_max)


def iterative_solver(A, b, lambda_max, epsilon=1e-3, max_iter=1000):
    n = A.shape[0]
    x = np.zeros(n)  # Начальное приближение
    tau = 1 / lambda_max  # Параметр tau
    for _ in range(max_iter):
        x_new = x - tau * (A @ x - b)
        if np.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new
    return x

x_iter = iterative_solver(A, b, lambda_max)
print("Решение системы (итерационный метод):", x_iter)



def jacobi_method(A, b, epsilon=1e-3, max_iter=1000):

    n = A.shape[0]  # Размерность системы
    x = np.zeros(n)  # Начальное приближение
    x_new = np.zeros(n)  # Вектор для нового приближения

    for iteration in range(max_iter):
        for i in range(n):
            # Вычисление суммы a_ij * x_j (используем только x из предыдущей итерации)
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]

            # Обновление x_new[i]
            x_new[i] = (b[i] - sigma) / A[i, i]

        # Критерий остановки: расстояние между текущим и новым приближением
        if np.linalg.norm(x_new - x) < epsilon:
            break
        # Обновление текущего приближения
        x = np.copy(x_new)

    return x

x_jacobi = jacobi_method(A, b)
print("Решение системы (метод Якоби):", x_jacobi)