import numpy as np 
import sympy as sp 
import math as mt 
import matplotlib.pyplot as plt 

class analitica():

    def calculate_analitica(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x, variancia):

        x = np.zeros(int(n_x) + 1)  # de 0 ao tamanho do reservatório com 10 elementos na malha
        t = np.zeros(int(n_t) + 1)  # de 0 a 10 segundos com 10 elementos

        if variancia == 'tempo':
            v = 'Steps de Tempo'
        elif variancia == 'malha':
            v = 'Malha'

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

        N = 200 # número de termos da série de Fourier

        # Cálculos Iniciais:
        def calculate_c2(k:float, rho:float, cp:float):

            c2 = k/(rho*cp)

            return c2

        c2 = calculate_c2(k,rho,cp)

        def calculate_lambda_n(c2: float, L:float):

            lambda_n = (np.sqrt(c2)*np.pi)/L

            return lambda_n

        lambda_n = calculate_lambda_n(c2,L)
        print(lambda_n)

        # Definição da Função Desejada:
        def f(x):
            return 20

        # Coeficiente da Série de Fourier:
        def calculate_Bn(f, L:float, N:float):

            Bn = np.zeros(N)
            for i in range(1,N):
                integrando = lambda x: f(x)*np.sin((i*np.pi*x)/L)
                Bn[i] = (2/L) * np.trapz(integrando(np.linspace(0,L,1000)), np.linspace(0,L,1000))

            return Bn

        # Solução da Equação do Calor:
        def calculate_calor(x:np.ndarray, t, N:float, L:float):

            calor = np.zeros_like(x)
            Bn = calculate_Bn(f,L,N)

            for i in range(1,N):
                calor += Bn[i]*np.sin((i*np.pi*x)/L) * np.exp(-(i*lambda_n**2)*t)

            return calor

        # Define valores de x e t:

        X, T = np.meshgrid(x,t)

        Temperatura = calculate_calor(X,T,N,L)
        print(Temperatura)

        # Plotagem dos Dados:

        # Plotagem:
        time = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(t)):
            if t[i] in time:
                plt.plot(x, Temperatura[i, :], linestyle='-', label=f't = {t[i]}')

        legend_label = f'{v} {n_x: .3f}' if variancia == "malha" else f'{v} {n_t: .3f}'
        plt.legend(labels=[legend_label])
        plt.title('Solução Analítica - Dirchlet')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Temperatura (°C)')
        plt.grid()
        plt.show()


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, T, Temperatura, rstride=2, cstride=2, cmap=plt.cm.viridis, linewidth=0.5, antialiased=True)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('t (s')
        ax.set_zlabel('T(x,y) (°C)')
        plt.title('Solução Analítica - Dirchlet')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        fig.text(0.02, 0.02, legend_label, color='black', ha='left')
        ax.view_init(30,60)

        plt.show()



        return X, t, Temperatura
