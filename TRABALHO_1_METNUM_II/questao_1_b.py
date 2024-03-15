# Resolução Analítica: Questão 1 - Letra B - Método de Euler

#Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

def calculate_letra_b():

    # Definindo Variáveis de Entrada:
    t0 = 0
    tf = 2
    y0 = 1

    h = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x = len(h) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf:float, t0: float, h:float) -> np.ndarray:

        n = np.zeros(len(h))
        for i in range(len(h)):

            n[i] = ((tf-t0)/(h[i])) +1 

        return n
    
    n = calculate_n(tf,t0,h)
    print(n) # vetor com os números de elementos de cada passo solicitado pelo problema 

    # Definindo a Função EDO:
    def f(t,y):
        return y*t**3 - 1.5*y

    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array = []
    y_array = []

    # Definindo as Condições Iniciais:
    for i in range(0,x): # loop que vai de 0 ao tamanho do vetor de passos

        t = np.linspace(t0,tf,int(n[i])) # cria um vetor de t, de t0 a tf com o número de elementos da interação (número de elementos no vetor, na posição i)
        y = np.zeros_like(t) # cria um vetor igual ao vetor criado para o tempo 
        y[0] = y0 # estabelece que na primeira posição do vetor de y, a condição inicial é y0
        y_array.append(y) # guarda os vetores de y dentro da matriz y_array
        t_array.append(t) # guarda os vetores de t dentro da matriz y_array

    print(t_array)
    print(y_array)

    # Método de Euler
    for i in range(x): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        m = int(n[i]) - 1 # vai mensurar o número de elementos menos 1 

        for j in range(0,m): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2
            y[j+1] = y[j] + h[i]*f(t[j], y[j]) # equacionamento do método de euler 
        y_array[i] = y # guarda o vetor y dentro matriz y_array

    print(y_array)

    # Plotagem da Solução Numérica:
    colors = ['#0000FF', '#1E90FF', '#4169E1', '#6495ED',   # Shades of blue
            '#87CEFA', '#ADD8E6', '#B0E0E6', '#87CEEB']   # More shades of blue

    for i in range(len(y_array)): # vai percorrer a matriz de y 

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t, y, marker='o', linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h[i]))
    
    plt.title('Letra B - Solução Numérica')
    plt.xlabel('Tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Definindo a Função da Solução Analítica:
    def calculate_yexata(t:np.ndarray) -> np.ndarray:

        y_exata_list = []

        for i in range(len(t)):
            
            y_exata = np.exp(((t[i]**4)/4) - 1.5*t[i])
            y_exata_list.append(y_exata)

        return y_exata_list

    y_exata_array = [] # cria uma matriz para inserir os vetores, com os valores de solução analítica, que vão ser gerados para cada vetor de t da matriz t_array

    for i in range(len(y_array)): # percorre a matriz de soluções analíticas y_array

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata = calculate_yexata(t) # calcula a solução exata com a função da solução analítica 
        y_exata_array.append(y_exata) # guarda o vetor y_exata dentro da matriz y_exata_array
 
    print('y_exata_array', y_exata_array)

    # Plotagem da Solução Analítica:
    colors = ['#006400', '#008000', '#556B2F',   # Shades of dark green
            '#8FBC8F', '#98FB98', '#90EE90',   # Shades of light green
            '#00FF00', '#7CFC00', '#32CD32']   # Shades of green

    for i in range(len(t_array)): # percorre a matriz de tempos t_array

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata = y_exata_array[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i 
        color = colors[i % len(colors)]
        plt.plot(t, y_exata, marker='o', linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h[i]))

    plt.title('Letra B - Solução Analítica')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Comparação das Soluções Analíticas e Numéricas:
    dark_pink = 'blue'
    light_pink = 'green'
    lilac = 'pink'
    purple = 'black'

    colors = [dark_pink, light_pink, lilac, purple]

    for i in range(len(t_array)): # percorre a matriz de tempos t_array

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata = y_exata_array[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        color = colors[i % len(colors)]
        plt.plot(t, y, marker='o', linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h[i]))
        plt.plot(t, y_exata, marker='o', linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h[i]))

    plt.title('Letra B - Comparação das Soluções Analítica e Numérica')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Cálculo do Erro Percentual Verdadeiro:
    def calculate_erro(y_exata:np.ndarray, y:np.ndarray) -> np.ndarray: # a função devolve uma lista/vetor

        erro_list = []
        for i in range(len(y_exata)): # percorre o vetor de y_exata 
            
            erro = np.abs((y_exata[i]-y[i])/y_exata[i])*100
            erro_list.append(erro) # guarda o valor dentro do vetor erro_list
        
        return erro_list

    erro_array = [] # cria uma matriz para guardar os vetores de erro_list correspondente a cada vetor de y_exata presente na matriz y_exata_array

    for i in range(len(y_array)): # percorre a matrzi de t_array 

        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata = y_exata_array[i]  # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        erro = calculate_erro(y_exata, y) # calcula o erro 
        erro_array.append(erro) # guarda o vetor de erros erro_list dentro da matriz erro_arrat

    print('erro', erro_array)

    # Plotagem do Erro Percentual Verdadeiro vs Tempo:
    colors = ['#FF4500', '#FF6347', '#FF7F50',   # Shades of dark orange
            '#FFA07A', '#FFDAB9', '#FFE4B5',   # Shades of light orange
            '#FF8C00', '#FFA500', '#FFD700']   # Shades of orange

    for i in range(len(t_array)): # percorre a matriz de t_array

        t = t_array[i]  # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        erro = erro_array[i]  # vai adotar o vetor de erro presente na matriz erro_array correspondente a posição i 
        color = colors[ i% len(colors)]
        plt.plot(t, erro, marker='o', linestyle='-', color=color, label='Erro Percentual Verdadeiro [%] - Passo ' + str(h[i]))

    plt.title('Letra B - Erro Percentual Verdadeiro vs Tempo')
    plt.xlabel('Tempo(s)')
    plt.ylabel('Erro Percentual Verdadeiro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()