# Resolução Analítica: Questão 1 - Letra A - Solução Analítica 

import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

def calculate_letra_a():

    #Definindo Intervalo de Tempo:
    t = [0,1,2] 

    # Definição da Função que Calcula a Solução Analítica:
    def calculate_yexata(t:float) -> float:

        y_exata = np.exp(((t[i]**4)/4) - 1.5*t[i])

        return y_exata 

    list_yexata = [] # cria uma lista para as soluções exatas de cada valor de t 

    for i in range(len(t)):

        y_exata = calculate_yexata(t) # chama a função 
        list_yexata.append(y_exata) # guarda o valor da interação dentro de uma lista 
        
    print(list_yexata)

    #Plotando a Solução Exata:
    plt.plot(t, list_yexata, marker='o', linestyle='-', color='#7B2791', label='Solução Analítica/Exata')

    plt.title('Letra A - Solução Analítica')
    plt.xlabel('tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

