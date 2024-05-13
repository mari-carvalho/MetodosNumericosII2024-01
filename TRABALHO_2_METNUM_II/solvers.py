import numpy as np

class solvers():

    def gauss_seidel(A, b, x0, Eppara, maxit):
        ne = len(
            b)  # medir o tamanho do vetor de P no tempo anterior mais 8/3*eta*rx*Pw(ou P0), número de equações do sistema
        x = np.zeros(ne) if x0 is None else np.array(
            x0)  # vetor de x com zeros se não for dado chute inicial, e vetor x0 se for fornecido chute inicial

        iter = 0
        Epest = np.linspace(100, 100, ne)  # vetor de erros relativos

        while np.max(
                Epest) >= Eppara and iter <= maxit:  # enquanto o valor máximo do vetor de erros for maior ou igual ao critério de parada e o número de interações for menor que o máximo
            x_old = np.copy(x)  # cria um vetor igual ao vetor de x, para guardar os valores encontrados e comparar

            for i in range(ne):  # loop nas equações do sistema
                sum1 = np.dot(A[i, :i], x[
                                        :i])  # soma dos produtos escalares dos coeficientes da linha i da matriz de A pelo valores de x até a posição i
                sum2 = np.dot(A[i, i + 1:], x_old[
                                            i + 1:])  # soma dos produtos escalares dos coeficientes da linha i da matriz A pelos valores de x_old após a posição i
                x[i] = (b[i] - sum1 - sum2) / A[i, i]  # atualiza x usando Gauss-Seidel

            Epest = np.abs((x - x_old) / x) * 100  # erro relativo

            iter += 1  # número de interações

        return x

    def tdma(A, D):
        a = np.diagonal(A, offset = -1)
        b = np.diagonal(A, offset = 0)
        c = np.diagonal(A, offset = +1)
        d = D

        n = len(d)
        c_ = np.zeros(n-1)
        d_ = np.zeros(n)
        x = np.zeros(n)

        # Forward elimination
        c_[0] = c[0]/b[0]
        d_[0] = d[0]/b[0]
        for i in range(1, n-1):
            c_[i] = c[i] / (b[i] - a[i-1] * c_[i-1])
        for i in range(1, n):
            d_[i] = (d[i] - a[i-1] * d[i-1]) / (b[i] - a[i-1] * c_[i-1])

        # Backward substitution
        x[-1] = d_[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i+1]

        return x

    def jacobi(A, b, x0, Eppara, maxit):
        ne = len(b)
        x = np.zeros(ne) if x0 is None else np.array(x0)

        iter = 0
        Epest = np.linspace(100,100,ne)
        while np.max(Epest) >= Eppara and iter <= maxit:
            x_old = np.copy(x)
            for i in range(ne):
                sum = np.dot(A[i, :i], x_old[:i]) + np.dot(A[i, i+1:], x_old[i + 1:])
                x[i] = (b[i] - sum)/A[i, i]

            Epest = np.abs((x - x_old)/x)*100

            iter += 1

        return x

    def gauss_seidel_relax(A, b, x0, Eppara, maxit, Lambda):
        ne = len(b)
        x = np.zeros(ne) if x0 is None else np.array(x0)

        iter = 0
        Epest = np.linspace(100,100,ne)

        while np.max(Epest) >= Eppara and iter <= maxit:
            x_old = np.copy(x)

            for i in range(ne):
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, i+1:], x_old[i +1:])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

            Epest = np.abs((x - x_old) / x) * 100

            iter += 1

            # Relaxamento
            x = Lambda*x + (1-Lambda)*x_old

        return x