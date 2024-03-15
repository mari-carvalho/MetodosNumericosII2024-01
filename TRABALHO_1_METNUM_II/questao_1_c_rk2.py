# Resolução Analítica: Questão 1 - Letra C - Solução Numérica com os Métodos de Runge-Kutta 2ª e 4ª Ordem

# Importando Bibliotecas:
import numpy as np
import math as mt 
import matplotlib.pyplot as plt


def calculate_letra_c_rk2():

    # Definindo Variáveis de Entrada:
    t0 = 0
    tf = 2
    y0 = 1
    h = 0.5

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf:float, t0:float, h:float) -> float:

        n = ((tf-t0)/h) + 1

        return n 

    n = calculate_n(tf,t0,h)
    print(n) # número de elementos 

    # Definindo a Função EDO:
    def f1(t,y): 
        return (y*(t**3)) - 1.5*y

    # Definindo os Vetores de Tempo e Y:
    t = np.zeros(int(n))
    y = np.zeros(int(n))

    # Definindo as Condições Iniciais:
    for i in range(0,int(n)):
        if i == 0:
            y[i] = y0 # na primeira posição i, 0, o valor de y deve ser o inicial 
            t[i] = t0 # na primeira posição i, 0, o valor de t deve ser o inicial 
        else:
            t[i] = i*h # se a posição for outra, que não a primeira, o tempo será contado de acordo com o passo 
            y[i] = 0 # as demais posições do vetor de y devem ser 0 

    # Método de Runge-Kutta 2ª Ordem:
    for i in range(0,int(n)-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2

        k1 = h*f1(t[i], y[i])
        k2 = h*f1(t[i+1], y[i] + k1)
        y[i+1] = y[i] + (1/2)*(k1 + k2) # método de runge-kutta 2ª ordem 

    print(y[i+1])

    # Definindo a Função da Solução Analítica:
    def calculate_yexata(t:np.ndarray) -> np.ndarray:

        y_exata_list = [] # cria uma lista para guardar os valores da solução exata de acordo com cada valor do vetor de t 

        for i in range(len(t)):
            
            y_exata = np.exp(((t[i]**4)/4) - 1.5*t[i])
            y_exata_list.append(y_exata) # guarda o valor dentro do vetor de y_exata_list

        return y_exata_list

    yex = calculate_yexata(t) # calcula a solução analítica

    # Plotagem da Comparação entre as Soluções Analítica e Numérica:
    plt.plot(t, y, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica')
    plt.plot(t, yex, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata')
    plt.title('Letra C - RK2 - Comparação das Soluções Analítica e Numérica')
    plt.xlabel('tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Definindo a Função que calcula o Erro Percentual Verdadeiro:
    def calculate_erro(yex:np.ndarray, y:np.ndarray) -> np.ndarray: # a função devolve uma lista/vetor

        erro_list = []
    
        for i in range(len(yex)): # percorre o vetor de y_exata 
            
            erro = np.abs((yex[i]-y[i])/yex[i])*100
            erro_list.append(erro) # guarda o valor dentro do vetor erro_list
        
        return erro_list
    
    erro = calculate_erro(yex, y)

    # Plotagem do Erro Percentual Verdadeiro vs Tempo:
    plt.plot(t, erro, marker='o', linestyle='-', color='red', label='Erro Percentual Verdadeiro [%]')
    plt.title('Letra C - RK2 - Erro Percentual Verdadeiro vs Tempo')
    plt.xlabel('tempo(s)')
    plt.ylabel('Erro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()