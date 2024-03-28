# Questão 1 - Letra A - Solução Analítica 

from sympy import symbols, Function, Eq, Derivative, exp, dsolve, simplify 
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

# Encontrando a Solução Analítica Numericamente:
# Definindo as Variáveis Simbólicas:
t = symbols('t')
y = Function('y')(t)

# Definindo a Equação Diferencial:

eq = Eq(Derivative(y,t), y * t**3 -1.5*y) # define a derivada de y em relação a t e em seguida a função a ser derivada 

# Encontrando a Solução Geral 

solucao_analitica = simplify(dsolve(eq, y))

print(solucao_analitica)

# Calculando a Solução Analítica e Plotando:
def calculate_letra_a():

    #Definindo Intervalo de Tempo:
    t = [0,1,2] 

    # Definição da Função que Calcula a Solução Analítica:
    def calculate_yexata(t:float) -> float:

        y_exata = np.exp(((t[i]**4)/4) - 1.5*t[i])

        return y_exata 

    y_exata_list = [] # cria uma lista para as soluções exatas de cada valor de t 

    for i in range(len(t)):

        y_exata = calculate_yexata(t) # chama a função 
        y_exata_list.append(y_exata) # guarda o valor da interação dentro de uma lista 
        
    print(y_exata_list)

    # Plotando a Solução Numérica/Exata:
    plt.plot(t, y_exata_list, linestyle='-', color='#7B2791', label='Solução Analítica/Exata')

    plt.title('Letra A - Solução Analítica - y(t) = e(t^4/4 - 1.5*t)')
    plt.xlabel('tempo(s)')
    plt.ylabel('y(t)')
    plt.grid()
    plt.legend()
    plt.show()

letra_a = calculate_letra_a()

