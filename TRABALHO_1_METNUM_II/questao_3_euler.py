# Resolução Analítica: Questão 3 - Sistema de EDO com Método de Euler:

#Importando Bibliotecas:
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def calculate_euler():

    # Definindo Variáveis de Entrada:
    n = 150
    tf = 10
    t0 = 0
    T0 = 16
    C0 = 1
    h = (tf-t0)/(n-1)

    # Definindo o Sistema de Funções de EDO:
    def f_C(t,T,C):
        return -np.exp(-10/(T+273)) * C

    def f_T(t,T,C):
        return 1000 * np.exp(-10/(T+273)) * C - 10*(T-20)

    # Definindo os Vetores de Tempo e de Y:
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

    # Método de Euler:
    for i in range(0,n-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2

        T[i+1] = T[i] + h*f_T(t[i], T[i], C[i]) # solução numérica da temperatura com o método de euler 
        C[i+1] = C[i] + h*f_C(t[i], T[i], C[i]) # solução numérica da composição com o método de euler 
    
    print('T', T)
    print('C', C)

    #FAREMOS A EXATA??
    #yex_v = np.sqrt((g*m)/cd) * np.tanh(np.sqrt((g*m)/cd)) 
    #yex_x = (m/cd) * (np.log(np.cosh(np.sqrt((g*cd)/m)*t)))

    # Plotagem da Solução do Sistema de EDO com o Método de Euler:
    # Solução Numérica da Temperatura:
    plt.plot(t, T, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica - Temperatura')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
    plt.title('Solução Numérica da Temperatura - Método de Euler')
    plt.xlabel('tempo(s)')
    plt.ylabel('T(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Solução Numérica da Composição:
    plt.plot(t, C, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica - Composição')
    #plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
    plt.title('Solução Numérica da Composição - Método de Euler')
    plt.xlabel('tempo(s)')
    plt.ylabel('C(t)')
    plt.grid(True)
    plt.legend()
    plt.show() 
