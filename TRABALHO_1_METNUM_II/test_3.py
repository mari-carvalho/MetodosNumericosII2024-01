# Resolução Analítica: Questão 1 - Letra B - Método de Euler

#Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 



def calculate_letra_b():

    # Definindo Variáveis de Entrada:
    t0 = 0
    tf = 10
    T0 = 16
    C0 = 1

    h = [0.009, 0.001] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x = len(h) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf:float, t0: float, h:float) -> np.ndarray:

        n = np.zeros(len(h))
        for i in range(len(h)):

            n[i] = ((tf-t0)/(h[i])) +1 

        return n
    
    n = calculate_n(tf,t0,h)
    print('n', n) # vetor com os números de elementos de cada passo solicitado pelo problema 

    # Definindo o Sistema de Funções de EDO:
    def f_C(t,T,C):
        return -np.exp(-10/(T+273)) * C

    def f_T(t,T,C):
        return 1000 * np.exp(-10/(T+273)) * C - 10*(T-20)

    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array = []
    T_array = []
    C_array = []

    # Definindo as Condições Iniciais:
    for i in range(0,x): # loop que vai de 0 ao tamanho do vetor de passos
        m = int(n[i])
        t = np.zeros(m)
        T = np.zeros(m)
        C = np.zeros(m)

        for j in range(0,m):
            if j == 0:
                t[j] = t0
                T[j] = T0
                C[j] = C0
            else:
                t[j] = j*h[i]
                T[j] = 0
                C[j] = 0
        T_array.append(T) # guarda os vetores de y dentro da matriz y_array
        C_array.append(C)
        t_array.append(t) # guarda os vetores de t dentro da matriz y_array

    print('t_array', t_array)
    print('T_array', T_array)
    print('C_array', C_array)

    for i in range(x):
        t = t_array[i]
        T = T_array[i]
        C = C_array[i]
        m = int(n[i]) - 1

        for j in range(0,m):
            T[j+1] = T[i] + h[i]*f_T(t[j], T[j], C[j]) # solução numérica da temperatura com o método de euler 
            C[j+1] = C[j] + h[i]*f_C(t[j], T[j], C[j]) # solução numérica da composição com o método de euler 
        T_array[i] = T
        C_array[i] = C
    
    print(T_array)
    print(C_array)

    # Plotagem da Solução Numérica:
    colors =  ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(t_array)):
        t = t_array[i]
        T = T_array[i]
        color = colors[i % len(colors)]
        plt.plot(t, T, marker='o', linestyle='-', color=color, label='Solução Numperica - Temperatura')
    plt.title('Solução Numérica da Temperatura - Método de Euler')
    plt.xlabel('tempo(s)')
    plt.ylabel('T(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    colors =  ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(t_array)):
        t = t_array[i]
        C = C_array[i]
        color = colors[i % len(colors)]
        plt.plot(t, T, marker='o', linestyle='-', color=color, label='Solução Numperica - Composição')
    plt.title('Solução Numérica da Temperatura - Método de Euler')
    plt.xlabel('tempo(s)')
    plt.ylabel('C(t)')
    plt.grid(True)
    plt.legend()
    plt.show()


b = calculate_letra_b()