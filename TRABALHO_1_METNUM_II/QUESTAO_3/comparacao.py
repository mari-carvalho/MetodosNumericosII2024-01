# Resolução Analítica: Questão 3 - Sistema de EDO com Método de Euler:

#Importando Bibliotecas:
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from prettytable import PrettyTable 

def calculate_comparacao():

    # Método de Euler------------------------------------
    # Definindo Variáveis de Entrada:
    n_euler = 150
    tf_euler = 10
    t0_euler = 0
    T0_euler = 16
    C0_euler = 1
    h_euler = (tf_euler-t0_euler)/(n_euler-1)

    # Definindo o Sistema de Funções de EDO:
    def f_C_euler(t_euler,T_euler,C_euler):
        return -np.exp(-10/(T_euler+273)) * C_euler

    def f_T_euler(t_euler,T_euler,C_euler):
        return 1000 * np.exp(-10/(T_euler+273)) * C_euler - 10*(T_euler-20)

    # Definindo os Vetores de Tempo e de Y:
    t_euler = np.zeros(n_euler)
    T_euler = np.zeros(n_euler)
    C_euler = np.zeros(n_euler)

    # Definindo as Condições Iniciais:
    for i in range(0,n_euler): # precisa de mais elementos que o numero de elementos, por causa do y[i+1]. Se for tempo de 0 a 4, 5 elementos, for precisa ir de 0 a 5 para dar 6 elementos 
        if i == 0:
            t_euler[i] = t0_euler
            T_euler[i] = T0_euler
            C_euler[i] = C0_euler
        else:
            t_euler[i] = i*h_euler 
            T_euler[i] = 0
            C_euler[i] = 0

    # Método de Euler:
    for i in range(0,n_euler-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2

        T_euler[i+1] = T_euler[i] + h_euler*f_T_euler(t_euler[i], T_euler[i], C_euler[i]) # solução numérica da temperatura com o método de euler 
        C_euler[i+1] = C_euler[i] + h_euler*f_C_euler(t_euler[i], T_euler[i], C_euler[i]) # solução numérica da composição com o método de euler 
    
    print('T', T_euler)
    print('C', C_euler)

    # Plotagem da Solução do Sistema de EDO com o Método de Euler:
    # Solução Numérica da Temperatura:
    plt.plot(t_euler, T_euler, linestyle='-', color='#FF1493', label='Solução Numérica - Temperatura')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
    plt.title('Solução Numérica da Temperatura por Euler')
    plt.xlabel('tempo(s)')
    plt.ylabel('T(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Solução Numérica da Composição:
    plt.plot(t_euler, C_euler, linestyle='-', color='#4B0082', label='Solução Numérica - Composição')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
    plt.title('Solução Numérica da Composição por Euler')
    plt.xlabel('tempo(s)')
    plt.ylabel('C(t)')
    plt.grid(True)
    plt.legend()
    plt.show() 

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_euler, "s"])
    tabela.add_row(["tf", tf_euler, "s"])
    tabela.add_row(["T0", T0_euler, "°C"])
    tabela.add_row(["C0", C0_euler, "mol/L"])
    tabela.add_row(["n", n_euler, "-"])
    tabela.add_row(["h", h_euler, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable(['t(s)', 'T(t)', 'C(t)'])
    for val1, val2, val3 in zip(t_euler, T_euler, C_euler):
            tabela.add_row([val1, val2, val3])

    print(tabela)

    # Método de rk2------------------------------------
    # Definindo Variáveis de Entrada:
    n_rk2 = 150
    tf_rk2 = 10
    t0_rk2 = 0
    T0_rk2 = 16
    C0_rk2 = 1
    h_rk2 = (tf_rk2-t0_rk2)/(n_rk2-1)

    # Definindo o Sistema de Funções de EDO:
    def f_C_rk2(t_rk2,T_rk2,C_rk2):
        return -np.exp(-10/(T_rk2+273)) * C_rk2

    def f_T_rk2(t_rk2,T_rk2,C_rk2):
        return 1000 * np.exp(-10/(T_rk2+273)) * C_rk2 - 10*(T_rk2-20)

    # Definindo os Vetores de Tempo e de Y:
    t_rk2 = np.zeros(n_rk2)
    T_rk2 = np.zeros(n_rk2)
    C_rk2 = np.zeros(n_rk2)

    # Definindo as Condições Iniciais:
    for i in range(0,n_rk2): # precisa de mais elementos que o numero de elementos, por causa do y[i+1]. Se for tempo de 0 a 4, 5 elementos, for precisa ir de 0 a 5 para dar 6 elementos 
        if i == 0:
            t_rk2[i] = t0_rk2
            T_rk2[i] = T0_rk2
            C_rk2[i] = C0_rk2
        else:
            t_rk2[i] = i*h_rk2 
            T_rk2[i] = 0
            C_rk2[i] = 0

    # Método de rk2:
    for i in range(0,n_rk2-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2

        k1_T_rk2 = h_rk2*f_T_rk2(t_rk2[i], T_rk2[i], C_rk2[i])
        k1_C_rk2 = h_rk2*f_C_rk2(t_rk2[i], T_rk2[i], C_rk2[i])
        k2_T_rk2 = h_rk2*f_T_rk2(t_rk2[i], T_rk2[i] + k1_T_rk2, C_rk2[i] + k1_C_rk2)
        k2_C_rk2 = h_rk2*f_C_rk2(t_rk2[i], T_rk2[i] + k1_T_rk2, C_rk2[i] + k1_C_rk2)
        T_rk2[i+1] = T_rk2[i] + (1/2)*(k1_T_rk2 + k2_T_rk2) # solução numérica da temperatura com o método de runge-kutta 2ª ordem
        C_rk2[i+1] = C_rk2[i] + (1/2)*(k1_C_rk2 + k2_C_rk2) # solução numérica da composição com o método de runge-kutta 2ª ordem
    
    print('T', T_rk2)
    print('C', C_rk2)

    # Plotagem da Solução do Sistema de EDO com o Método de rk2:
    # Solução Numérica da Temperatura:
    plt.plot(t_rk2, T_rk2, linestyle='-', color='#FF1493', label='Solução Numérica - Temperatura')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
    plt.title('Solução Numérica da Temperatura por rk2')
    plt.xlabel('tempo(s)')
    plt.ylabel('T(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Solução Numérica da Composição:
    plt.plot(t_rk2, C_rk2, linestyle='-', color='#4B0082', label='Solução Numérica - Composição')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
    plt.title('Solução Numérica da Composição por rk2')
    plt.xlabel('tempo(s)')
    plt.ylabel('C(t)')
    plt.grid(True)
    plt.legend()
    plt.show() 

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk2, "s"])
    tabela.add_row(["tf", tf_rk2, "s"])
    tabela.add_row(["T0", T0_rk2, "°C"])
    tabela.add_row(["C0", C0_rk2, "mol/L"])
    tabela.add_row(["n", n_rk2, "-"])
    tabela.add_row(["h", h_rk2, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable(['t(s)', 'T(t)', 'C(t)'])
    for val1, val2, val3 in zip(t_rk2, T_rk2, C_rk2):
            tabela.add_row([val1, val2, val3])

    print(tabela)

    
    # Método de rk4------------------------------------
    # Definindo Variáveis de Entrada:
    n_rk4 = 150
    tf_rk4 = 10
    t0_rk4 = 0
    T0_rk4 = 16
    C0_rk4 = 1
    h_rk4 = (tf_rk4-t0_rk4)/(n_rk4-1)

    # Definindo o Sistema de Funções de EDO:
    def f_C_rk4(t_rk4,T_rk4,C_rk4):
        return -np.exp(-10/(T_rk4+273)) * C_rk4

    def f_T_rk4(t_rk4,T_rk4,C_rk4):
        return 1000 * np.exp(-10/(T_rk4+273)) * C_rk4 - 10*(T_rk4-20)

    # Definindo os Vetores de Tempo e de Y:
    t_rk4 = np.zeros(n_rk4)
    T_rk4 = np.zeros(n_rk4)
    C_rk4 = np.zeros(n_rk4)

    # Definindo as Condições Iniciais:
    for i in range(0,n_rk4): # precisa de mais elementos que o numero de elementos, por causa do y[i+1]. Se for tempo de 0 a 4, 5 elementos, for precisa ir de 0 a 5 para dar 6 elementos 
        if i == 0:
            t_rk4[i] = t0_rk4
            T_rk4[i] = T0_rk4
            C_rk4[i] = C0_rk4
        else:
            t_rk4[i] = i*h_rk4 
            T_rk4[i] = 0
            C_rk4[i] = 0

    # Método de rk4:
    for i in range(0,n_rk4-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2

        k1_T_rk4 = h_rk4*f_T_rk4(t_rk4[i], T_rk4[i],  C_rk4[i])
        k1_C_rk4 = h_rk4*f_C_rk4(t_rk4[i], T_rk4[i], C_rk4[i])
        k2_T_rk4 = h_rk4*f_T_rk4(t_rk4[i] + (h_rk4/2), T_rk4[i] + (k1_T_rk4/2),  C_rk4[i] + (k1_C_rk4/2))
        k2_C_rk4 = h_rk4*f_C_rk4(t_rk4[i] + (h_rk4/2), T_rk4[i] + (k1_T_rk4/2), C_rk4[i] + (k1_C_rk4/2))
        k3_T_rk4 = h_rk4*f_T_rk4(t_rk4[i] + (h_rk4/2), T_rk4[i] + (k2_T_rk4/2), C_rk4[i] + (k2_C_rk4/2))
        k3_C_rk4 = h_rk4*f_C_rk4(t_rk4[i] + (h_rk4/2), T_rk4[i] + (k2_T_rk4/2), C_rk4[i] + (k2_C_rk4/2))
        k4_T_rk4 = h_rk4*f_T_rk4(t_rk4[i] + h_rk4, T_rk4[i] + k3_T_rk4, C_rk4[i] + k3_C_rk4)
        k4_C_rk4 = h_rk4*f_C_rk4(t_rk4[i] + h_rk4, T_rk4[i] + k3_T_rk4, C_rk4[i] + k3_C_rk4)
        T_rk4[i+1] = T_rk4[i] + (1/6)*(k1_T_rk4 + (2*k2_T_rk4) + (2*k3_T_rk4) + k4_T_rk4) # solução numérica da temperatura com o método de runge-kutta 4ª ordem 
        C_rk4[i+1] = C_rk4[i] + (1/6)*(k1_C_rk4 + (2*k2_C_rk4) + (2*k3_C_rk4) + k4_C_rk4) # solução numérica da composição com o método de runge-kutta 4ª ordem
    
    print('T', T_rk4)
    print('C', C_rk4)

    # Plotagem da Solução do Sistema de EDO com o Método de rk4:
    # Solução Numérica da Temperatura:
    plt.plot(t_rk4, T_rk4, linestyle='-', color='#FF1493', label='Solução Numérica - Temperatura')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
    plt.title('Solução Numérica da Temperatura por rk4')
    plt.xlabel('tempo(s)')
    plt.ylabel('T(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Solução Numérica da Composição:
    plt.plot(t_rk4, C_rk4, linestyle='-', color='#4B0082', label='Solução Numérica - Composição')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
    plt.title('Solução Numérica da Composição por rk4')
    plt.xlabel('tempo(s)')
    plt.ylabel('C(t)')
    plt.grid(True)
    plt.legend()
    plt.show() 

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk4, "s"])
    tabela.add_row(["tf", tf_rk4, "s"])
    tabela.add_row(["T0", T0_rk4, "°C"])
    tabela.add_row(["C0", C0_rk4, "mol/L"])
    tabela.add_row(["n", n_rk4, "-"])
    tabela.add_row(["h", h_rk4, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable(['t(s)', 'T(t)', 'C(t)'])
    for val1, val2, val3 in zip(t_rk4, T_rk4, C_rk4):
            tabela.add_row([val1, val2, val3])

    print(tabela)

    # Plotagem da Comparação das Soluções Numéricas:
    plt.plot(t_euler, T_euler, linestyle='-', color='#FF1493', label='Solução Numérica - Euler  - Temperatura')
    plt.plot(t_euler, T_rk2, linestyle='-', color='#4B0082', label='Solução Numérica - RK2  - Temperatura')
    plt.plot(t_euler, T_rk4, linestyle='-', color='#006400', label='Solução Numérica - RK4  - Temperatura')
    plt.title('Comparação das Soluções Numéricas')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(t_euler, C_euler, linestyle='-', color='#FF1493', label='Solução Numérica - Euler - Composição')
    plt.plot(t_euler, C_rk2, linestyle='-', color='#4B0082', label='Solução Numérica - RK2 - Composição')
    plt.plot(t_euler, C_rk4, linestyle='-', color='#006400', label='Solução Numérica - RK4 - Composição')
    plt.title('Comparação das Soluções Numéricas')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()


comp = calculate_comparacao()