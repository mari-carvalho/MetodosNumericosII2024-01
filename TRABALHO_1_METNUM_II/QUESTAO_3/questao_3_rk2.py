# Resolução Analítica: Questão 3 - Sistema de EDO com Método de Runge-Kutta 2ª Ordem:

#Importando Bibliotecas:
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from prettytable import PrettyTable 

def calculate_3_rk2():

    # Definindo Variáveis de Entrada:
    n = 150
    tf = 10
    t0 = 0
    T0 = 16
    C0 = 1
    h = (tf-t0)/(n-1)

    # Definindo o Sistema de Funções de EDO:
    def f_C(t,T,C):
        return - np.exp(-10/(T+273)) * C

    def f_T(t,T,C):
        return 1000 * np.exp(-10/(T+273)) * C - 10*(T-20)

    # Definindo os Vetores de Tempor e de Y:
    t = np.zeros(n)
    T = np.zeros(n)
    C = np.zeros(n)

    # Definindo as Condições Iniciais:
    for i in range(0,n): # precisa de mais elementos que o numero de elementos, por causa do y[i+1]. Se for tempo de 0 a 4, 5 elementos, for precisa ir de 0 a 5 para dar 6 elementos 
        if i == 0:
            t[i] = t0
            T[i] = T0
            C[i] = C0
        else:
            t[i] = i*h 
            T[i] = 0
            C[i] = 0

    # Método de Ruge-Kutta de 2ª Ordem:
    for i in range(0,n-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2

        k1_T = h*f_T(t[i], T[i], C[i])
        k1_C = h*f_C(t[i], T[i], C[i])
        k2_T = h*f_T(t[i], T[i] + k1_T, C[i] + k1_C)
        k2_C = h*f_C(t[i], T[i] + k1_T, C[i] + k1_C)
        T[i+1] = T[i] + (1/2)*(k1_T + k2_T) # solução numérica da temperatura com o método de runge-kutta 2ª ordem
        C[i+1] = C[i] + (1/2)*(k1_C + k2_C) # solução numérica da composição com o método de runge-kutta 2ª ordem

    print('T', T) 
    print('C', C)

    # Plotagem da Solução do Sistema de EDO com o Método de Runge-Kutta de 2ª Ordem:
    # Solução Numérica da Temperatura:
    plt.plot(t, T, linestyle='-', color='#FF1493', label='Temperatura')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
    plt.title('Solução Numérica da Temperatura - RK2')
    plt.xlabel('Tempo(s)')
    plt.ylabel('T(t) [°C]')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Solução Numérica da Composição:
    plt.plot(t, C, linestyle='-', color='#4B0082', label='Composição')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
    plt.title('Solução Numérica da Composição - RK2')
    plt.xlabel('Tempo(s)')
    plt.ylabel('C(t) [mol/L]')
    plt.grid(True)
    plt.legend()
    plt.show() 

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parametros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0, "s"])
    tabela.add_row(["tf", tf, "s"])
    tabela.add_row(["T0", T0, "°C"])
    tabela.add_row(["C0", C0, "mol/L"])
    tabela.add_row(["n", n, "-"])
    tabela.add_row(["h", h, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable(['t(s)', 'T(t)', 'C(t) [mol/L]'])
    for val1, val2, val3 in zip(t, T, C):
            tabela.add_row([val1, val2, val3])

    print(tabela)

a = calculate_3_rk2()
