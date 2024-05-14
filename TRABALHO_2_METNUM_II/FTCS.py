import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp

class FTCS():

    def calculate_FTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x):

        x = np.zeros(int(n_x) + 1)  # de 0 ao tamanho do reservatório com 10 elementos na malha
        t = np.zeros(int(n_t) + 1)  # de 0 a 10 segundos com 10 elementos
        T = np.zeros((int(n_t) + 1, int(n_x) + 1))
        tam = len(x)
        print('tam_ft', tam)

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
                print(i)
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
        print('T', T)

        def calculate_eta(k, rho, cp) -> float:
            eta = k / (rho*cp)

            return eta

        eta = calculate_eta(k, rho, cp)

        def calculate_rx(h_t: float, h_x: float) -> float:
            rx = (h_t) / (h_x ** 2)

            return rx

        rx = calculate_rx(h_t, h_x)

        def calculate_rxn(rx, eta):
            rxn = rx*eta

            return rxn

        rxn_tt = calculate_rxn(rx, eta)

        # Criando o método MDF de FTCS:

        for i in range(len(t)):  # varre as linhas
            for j in range(len(x)):  # varre as colunas
                if i == 0:  # tempo zero
                    T[i, j] = T0
                else:  # tempo diferente de zero
                    if j == 0:
                        T[i, j] = Te
                    else:
                        if j == 1:  # bloco 1
                            T[i, j] = (8 / 3) * eta * rx * Te + (1 - 4 * eta * rx) * T[i - 1, j] + (4 / 3) * eta * rx * T[i - 1, j + 1]
                        elif j == len(x) - 2:  # bloco N
                            T[i, j] = (4 / 3) * eta * rx * T[i - 1, j - 1] + (1 - 4 * eta * rx) * T[i - 1, j] + (
                                        8 / 3) * eta * rx * Tw
                        elif j == len(x) - 1:  # N+1
                            T[i, j] = Tw
                        else:  # blocos interiores
                            T[i, j] = eta * rx * T[i - 1, j - 1] + (1 - 2 * eta * rx) * T[i - 1, j] + eta * rx * T[i - 1, j + 1]

        print(T)

        # Plotagem:
        time = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(t)):
            if t[i] in time:
                plt.plot(x, T[i, :], linestyle='-', label=f't = {t[i]}')

        plt.legend()
        plt.title('Formulação FTCS - Dirchlet')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.grid()
        plt.show()

        return x, t, T

    def calculate_FTCS_tf(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, i, j, n_t, n_x):


        x = np.zeros(int(n_x)+1) # de 0 ao tamanho do reservatório com 10 elementos na malha 
        t = np.zeros(int(n_t)+1) # de 0 a 10 segundos com 10 elementos
        T = np.zeros((int(n_t)+1,int(n_x)+1))

        h_t = i
        h_x = j

        # Alimentando os vetores:
        for i in range(len(x)):
            if i == 0:
                x[i] = x0
            elif i == 1:
                x[i] = i*(h_x/2)
            elif i == len(x):
                x[i] = L 
            elif i == len(x)-1:
                x[i] = x[i-1] + (h_x/2)
            else:
                x[i] = x[i-1] + h_x
                
        for i in range(len(t)):
            if i == 0:
                t[i] = t0
            elif i == len(t):
                t[i] = tf
            else:
                t[i] = i*h_t      

        print('x', x)
        print('t', t)
        print('T', T)

        def calculate_eta(k, rho, cp) -> float:
            eta = k / (rho*cp)

            return eta

        eta = calculate_eta(k, rho, cp)

        def calculate_rx(h_t:float, h_x:float) -> float:
            rx = (h_t)/(h_x**2)

            return rx 

        rx = calculate_rx(h_t, h_x)

        def calculate_rxn(rx, eta):
            rxn = rx * eta

            return rxn

        rxn_tf = calculate_rxn(rx, eta)

        # Criando o método MDF de FTCS:

        for i in range(len(t)): # varre as linhas
            for j in range(len(x)): # varre as colunas 
                if i == 0: # tempo zero 
                    T[i,j] = T0
                else: # tempo diferente de zero 
                    if j == 0:
                        T[i,j] = T0
                    else:
                        if j == 1: # bloco 1
                            T[i, j] = (8 / 3) * eta * rx * Te + (1 - 4 * eta * rx) * T[i - 1, j] + (4 / 3) * eta * rx * T[i - 1, j + 1]
                        elif j == len(x)-2: # bloco N 
                            T[i,j] = eta*rx*T[i-1,j-1] + (1-eta*rx)*T[i-1,j] - eta*rx*((qw)/(h_x*k))
                        elif j == len(x)-1: # N+1
                            T[i,j] = T0
                        else: # blocos interiores 
                            T[i,j] = eta*rx*T[i-1,j-1] + (1-2*eta*rx)*T[i-1,j] + eta*rx*T[i-1,j+1]

        print(T)
                    
        # Plotagem:
        time = [0,10,20,30,40,50,60,70,80,90,100]
        for i in range(len(t)):
            if t[i] in time:
                plt.plot(x, t[i, :], linestyle='-', label=f't = {t[i]}')

        plt.legend()
        plt.title('Formulação FTCS - Neumann')
        plt.xlabel('Comprimento (m)')
        plt.ylabel('Pressão (psia)')
        plt.grid()
        plt.show()

        return x, t, t
