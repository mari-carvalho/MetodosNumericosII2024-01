from scipy.linalg import solve


class BTCS():

    def calculate_BTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x):

        import numpy as np
        import matplotlib.pyplot as plt
        import sympy as sp
        from solver_gauss_seidel import gauss_seidel

        x = np.zeros(int(n_x) + 1)  # de 0 ao tamanho do reservatório com 10 elementos na malha
        t = np.zeros(int(n_t) + 1)  # de 0 a 10 segundos com 10 elementos

        h_t = i
        h_x = j

        # Alimentando os vetores:
        for i in range(len(x)):
            if i == 0:
                x[i] = x0
            elif i == 1:
                x[i] = i * (h_x / 2)
            elif i == len(x):
                x[i] = L
            elif i == len(x) - 1:
                x[i] = x[i - 1] + (h_x / 2)
            else:
                x[i] = x[i - 1] + h_x

        for i in range(len(t)):
            if i == 0:
                t[i] = t0
            elif i == len(t):
                t[i] = tf
            else:
                t[i] = i * h_t

        print('x', x)
        print('t', t)

        def calculate_eta(k:float, rho:float, cp:float) -> float:
            eta = k / (rho*cp)

            return eta

        eta = calculate_eta(k, rho, cp)

        def calculate_rx(h_t: float, h_x: float) -> float:
            rx = (h_t) / (h_x ** 2)

            return rx

        rx = calculate_rx(h_t, h_x)

        # Criando o método MDF de BTCS:

        # Critérios de Scarvorought, 1966:
        n = 12
        Eppara = 0.5 * 10 ** -n

        # Número Máximo de Interações:
        maxit = 1000

        ai = -eta * rx
        bi = 1 + 2 * rx * eta
        an = -(4 / 3) * rx * eta
        b1 = 1 + 4 * rx * eta

        t_coeficientes = np.zeros((int(n_x) + 1, int(n_x) + 1))  # a matriz de coeficientes deve ser quadrada
        t_old = np.ones(int(n_x) + 1) * T0  # vai atualizar cada linha da matriz
        t_solucoes = np.zeros((int(n_t) + 2,int(n_x) + 1))  # matriz de soluções pode seguir a mesma lógica da FTCS, não quadrada. a matriz de soluções deve ter dimensão de 402 em linhas, porque o vetor de h começa em 0 + 1 = 1 (linha 0 da matriz de soluções é a condição inicial p0), para ir até 401 deve ter uma dimensão a mais (21 linhas, preenche até 20 começando de 0; 401 linhas, preenche até 400 começando de 0; 402 linhas, preenche até 400, começando de 1 )
        d = np.zeros(int(n_x) + 1)  # vai guardar os valores de p no tempo anterior mais 8/3*eta*rx*(Pw ou P0)
        h = 0  # para acompanhar o tamanho do vetor de tempo (0 a 9, 10 elementos), p_soluções deve ter uma posição a frente (1 a 9, 9 elementos)
        t_solucoes[h, :] = T0  # primeira linhas todas as colunas, tempo = 0

        for j in range(len(t)):  # 0 a 4 (tamanho de t), tempo 4 elemento 5; 1 a 4 (tamanho de t, mesmo for), tempo 4 elemento 4; precisa de mais um elemento
            h = h + 1
            t_coeficientes = np.zeros((int(n_x) + 1, int(n_x) + 1))
            for i in range(len(t_coeficientes)):  # variando a linha
                if i == 0:
                    t_coeficientes[i, 0] = b1
                    t_coeficientes[i, 1] = an
                    d[i] = t_old[i] + 8 / 3 * rx * eta * Te
                elif i == len(t_coeficientes) - 1:  # o último, N
                    t_coeficientes[i, len(t_coeficientes) - 2] = an
                    t_coeficientes[i, len(t_coeficientes) - 1] = b1
                    d[i] = t_old[i] + 8 / 3 * rx * eta * Tw
                else:
                    t_coeficientes[i, i - 1] = ai  # linha 1, coluna 0 (i-1)
                    t_coeficientes[i, i] = bi
                    t_coeficientes[i, i + 1] = ai
                    d[i] = t_old[i]  # condição central é 0

            x0 = t_old  # os primeiros valores de chute inicial vão ser os valores de p calculadas no tempo anterior
            t_new = gauss_seidel(t_coeficientes, d, x0, Eppara, maxit)
            # p_new = solve(p_coeficientes,d)
            t_old = t_new  # atualiza a matriz, coloca o vetor de pressões calculado no tempo anterior (p_new) em p_old
            #t_new = np.insert(t_new, 0, Tw)  # inserindo colunas
            #t_new = np.append(t_new, T0)  # append sempre no final
            t_solucoes[h,:] = t_new  # vai guardar na matriz de solucoes todos os vetores de pressão calculados nos tempos

        print(t_solucoes)
        print(t_coeficientes)
        tam1 = len(t_solucoes[0])
        tam2 = len(t_coeficientes)
        print('tam', tam1)
        print('tam2', tam2)

        # Plotagem:
        time = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(t)):
            if t[i] in time:
                plt.plot(x, t_solucoes[i, :], linestyle='-', label=f't = {t[i]}')

        plt.legend()
        plt.title('Formulação BTCS - Dirchlet')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.grid()
        plt.show()

        return x, t, t_solucoes
    
    def calculate_BTCS_tf(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x):

        import numpy as np
        import matplotlib.pyplot as plt
        import sympy as sp
        from solver_gauss_seidel import gauss_seidel

        x = np.zeros(int(n_x) + 1)  # de 0 ao tamanho do reservatório com 10 elementos na malha
        t = np.zeros(int(n_t) + 1)  # de 0 a 10 segundos com 10 elementos

        h_t = i
        h_x = j

        # Alimentando os vetores:
        for i in range(len(x)):
            if i == 0:
                x[i] = x0
            elif i == 1:
                x[i] = i * (h_x / 2)
            elif i == len(x):
                x[i] = L
            elif i == len(x) - 1:
                x[i] = x[i - 1] + (h_x / 2)
            else:
                x[i] = x[i - 1] + h_x

        for i in range(len(t)):
            if i == 0:
                t[i] = t0
            elif i == len(t):
                t[i] = tf
            else:
                t[i] = i * h_t

        print('x', x)
        print('t', t)

        def calculate_eta(k:float, rho:float, cp:float) -> float:
            eta = k / (rho*cp)

            return eta

        eta = calculate_eta(k, rho, cp)

        def calculate_rx(h_t: float, h_x: float) -> float:
            rx = (h_t) / (h_x ** 2)

            return rx

        rx = calculate_rx(h_t, h_x)

        # Criando o método MDF de BTCS:

        # Critérios de Scarvorought, 1966:
        n = 12
        Eppara = 0.5 * 10 ** -n

        # Número Máximo de Interações:
        maxit = 1000

        ai = -eta * rx
        bi = 1 + 2 * rx * eta
        an = -(4 / 3) * rx * eta
        b1 = 1 + 4 * rx * eta

        t_coeficientes = np.zeros((int(n_x) + 1, int(n_x) + 1))  # a matriz de coeficientes deve ser quadrada
        t_old = np.ones(int(n_x) + 1) * T0  # vai atualizar cada linha da matriz
        t_solucoes = np.zeros((int(n_t) + 2,int(n_x) + 1))  # matriz de soluções pode seguir a mesma lógica da FTCS, não quadrada. a matriz de soluções deve ter dimensão de 402 em linhas, porque o vetor de h começa em 0 + 1 = 1 (linha 0 da matriz de soluções é a condição inicial p0), para ir até 401 deve ter uma dimensão a mais (21 linhas, preenche até 20 começando de 0; 401 linhas, preenche até 400 começando de 0; 402 linhas, preenche até 400, começando de 1 )
        d = np.zeros(int(n_x) + 1)  # vai guardar os valores de p no tempo anterior mais 8/3*eta*rx*(Pw ou P0)
        h = 0  # para acompanhar o tamanho do vetor de tempo (0 a 9, 10 elementos), p_soluções deve ter uma posição a frente (1 a 9, 9 elementos)
        t_solucoes[h, :] = T0  # primeira linhas todas as colunas, tempo = 0

        for j in range(len(t)):  # 0 a 4 (tamanho de t), tempo 4 elemento 5; 1 a 4 (tamanho de t, mesmo for), tempo 4 elemento 4; precisa de mais um elemento
            h = h + 1
            t_coeficientes = np.zeros((int(n_x) + 1, int(n_x) + 1))
            for i in range(len(t_coeficientes)):  # variando a linha
                if i == 0:
                    t_coeficientes[i, 0] = b1
                    t_coeficientes[i, 1] = an
                    d[i] = t_old[i] + 8 / 3 * rx * eta * Te
                elif i == len(t_coeficientes) - 1:  # o último, N
                    t_coeficientes[i, len(t_coeficientes) - 2] = an
                    t_coeficientes[i, len(t_coeficientes) - 1] = b1
                    d[i] = t_old[i] + 8 / 3 * rx * eta * Te
                else:
                    t_coeficientes[i, i - 1] = ai  # linha 1, coluna 0 (i-1)
                    t_coeficientes[i, i] = bi
                    t_coeficientes[i, i + 1] = ai
                    d[i] = t_old[i]  # condição central é 0

            x0 = t_old  # os primeiros valores de chute inicial vão ser os valores de p calculadas no tempo anterior
            t_new = gauss_seidel(t_coeficientes, d, x0, Eppara, maxit)
            # p_new = solve(p_coeficientes,d)
            t_old = t_new  # atualiza a matriz, coloca o vetor de pressões calculado no tempo anterior (p_new) em p_old
            t_new = np.insert(t_new, 0, t_new[0] - (((qw*mi)/(k*A))*(h_x/2))) # inserindo colunas
            t_new = np.append(t_new, T0) # append sempre no final
            t_solucoes[h,:] = t_new  # vai guardar na matriz de solucoes todos os vetores de pressão calculados nos tempos

        print(t_solucoes)
        print(t_coeficientes)
        tam1 = len(t_solucoes[0])
        tam2 = len(t_coeficientes)
        print('tam', tam1)
        print('tam2', tam2)

        # Plotagem:
        time = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(t)):
            if t[i] in time:
                plt.plot(x, t_solucoes[i, :], linestyle='-', label=f't = {t[i]}')

        plt.legend()
        plt.title('Formulação BTCS - Dirchlet')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.grid()
        plt.show()

        return x, t, t_solucoes