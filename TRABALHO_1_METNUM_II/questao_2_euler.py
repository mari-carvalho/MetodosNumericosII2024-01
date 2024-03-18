# Resolução Analítica: Questão 2 - Equação de EDO com Método de Euler:

#Importando Bibliotecas:
import numpy as np
import math as mt
import matplotlib.pyplot as plt

def calculate_2_euler():

    # Definindo Variáveis de Entrada:
    g = 9.81 # m/s²
    d_orificio = 0.03 # m 
    r = 1.5
    h0 = 2.75 # m 
    C = 0.55 
    t0 = 0 
    tf = 1000
    Eppara = 10**(-6)
    
    p = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x = len(p) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf:float, t0: float, p:float) -> np.ndarray:

        n = np.zeros(len(p))
        for i in range(len(p)):

            n[i] = ((tf-t0)/(p[i])) +1 

        return n
    
    n = calculate_n(tf,t0,p)
    print(n) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area(d_orificio:float) -> float: 
        
        area = (np.pi/4) * d_orificio**2

        return area
    area = calculate_area(d_orificio)   
        
    # Definindo a Função EDO:
    def f(t,h):
        return (-C*area*np.sqrt(2*g))/np.sqrt(h)*(2*np.pi*r-np.pi*h)

    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array = []
    h_array = []

    # Definindo as Condições Iniciais:
    for i in range(0,x): # loop que vai de 0 ao tamanho do vetor de passos

        t = np.linspace(t0,tf,int(n[i])) # cria um vetor de t, de t0 a tf com o número de elementos da interação (número de elementos no vetor, na posição i)
        h = np.zeros_like(t) # cria um vetor igual ao vetor criado para o tempo 
        h[0] = h0 # estabelece que na primeira posição do vetor de y, a condição inicial é y0
        h_array.append(h) # guarda os vetores de y dentro da matriz y_array
        t_array.append(t) # guarda os vetores de t dentro da matriz y_array

    print(t_array)
    print(h_array)

    # Método de Euler
    for i in range(x): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        m = int(n[i]) - 1 # vai mensurar o número de elementos menos 1 

        for j in range(0,m): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2
            h[j+1] = h[j] + p[i]*f(t[j], h[j]) # equacionamento do método de euler 
        h_array[i] = h # guarda o vetor y dentro matriz y_array

    print(h_array)

    # Plotagem da Solução Numérica:
    colors = ['#0000FF', '#1E90FF', '#4169E1', '#6495ED',   # Shades of blue
            '#87CEFA', '#ADD8E6', '#B0E0E6', '#87CEEB']   # More shades of blue

    for i in range(len(h_array)): # vai percorrer a matriz de y 

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t, h, marker='o', linestyle='-', color=color, label='Solução Numérica - Passo ' + str(p[i]))
    
    plt.title('Letra B - Solução Numérica')
    plt.xlabel('Tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

euler2 = calculate_2_euler()
print(euler2)
