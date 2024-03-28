# Resolução Analítica: Questão 1 - Letra C - Método de Runge-Kutta de 2º ordem

#Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 
from prettytable import PrettyTable 

def calculate_letra_c_rk2():

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

            k1 = h[i]*f(t[j], y[j])
            k2 = h[i]*f(t[j+1], y[j] + k1)
            y[j+1] = y[j] + (1/2)*(k1 + k2) # método de runge-kutta 2ª ordem 
        y_array[i] = y # guarda o vetor y dentro matriz y_array

    print(y_array)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(t_array)): # vai percorrer a matriz de y 

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t, y, linestyle='-', color=color, label='Passo ' + str(h[i]))
    
    plt.title('Letra C - Solução Numérica por RK2 - dy/dt = y*t³ - 1.5*y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Definindo a Função da Solução Analítica:
    def calculate_yexata(t:np.ndarray) -> np.ndarray:

        y_exata_list = []

        for i in range(len(t)):
            
            y_exata = np.exp(t[i]*(0.25*t[i]**3 - 1.5))
            y_exata_list.append(y_exata)

        return y_exata_list

    y_exata_array = [] # cria uma matriz para inserir os vetores, com os valores de solução analítica, que vão ser gerados para cada vetor de t da matriz t_array

    for i in range(len(t_array)): # percorre a matriz de soluções analíticas y_array

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata = calculate_yexata(t) # calcula a solução exata com a função da solução analítica 
        y_exata_array.append(y_exata) # guarda o vetor y_exata dentro da matriz y_exata_array
 
    print('y_exata_array', y_exata_array)

    # Plotagem da Solução Analítica:
    colors = ['#006400', '#7CFC00']   # Dark green, medium spring green

    for i in range(len(t_array)): # percorre a matriz de tempos t_array

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata = y_exata_array[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i 
        color = colors[i % len(colors)]
        plt.plot(t, y_exata, linestyle='-', color=color, label='Passo ' + str(h[i]))

    plt.title('Letra C - Solução Analítica - y(t) = e(t^4/4 - 1.5*t)')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Comparação das Soluções Analíticas e Numéricas:
    colors_numerica = '#006400'
    colors_analitica = '#FF1493'

    for i in range(len(t_array)): # percorre a matriz de tempos t_array

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata = y_exata_array[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        print('y_exata', y_exata)
        plt.plot(t, y, linestyle='-', color=colors_numerica, label='Passo ' + str(h[i]))
        plt.plot(t, y_exata, linestyle='-', color=colors_analitica, label='Passo ' + str(h[i]))

    plt.title('Letra C - Comparação das Soluções Analítica e Numérica')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Cálculo do Erro Percentual Verdadeiro:
    def calculate_erro(y_exata:np.ndarray, y:np.ndarray, t:np.ndarray) -> np.ndarray: # a função devolve uma lista/vetor

        erro_list = []

        for j in range(len(y_exata)): # percorre o vetor de y_exata 
            print('y_exata_func', y_exata)
            print('y_func', y)
            erro = np.abs((y_exata[j]-y[j])/y_exata[j])*100
            print('erro_func', erro)
            erro_list.append(erro) # guarda o valor dentro do vetor erro_list
        
        return erro_list

    erro_array = [] # cria uma matriz para guardar os vetores de erro_list correspondente a cada vetor de y_exata presente na matriz y_exata_array

    for i in range(len(t_array)): # percorre a matriz de t_array 

        t = t_array[i]
        y = y_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        print('y_essa', y)
        y_exata = y_exata_array[i]  # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        print('y_exata_essa', y_exata)
        erro = calculate_erro(y, y_exata, t) # calcula o erro 
        erro_array.append(erro) # guarda o vetor de erros erro_list dentro da matriz erro_arrat

    print('erro', erro_array)

    # Plotagem do Erro Percentual Verdadeiro vs Tempo:
    colors = ['#FF4500', '#FF8C00']   # Dark orange, orange

    for i in range(len(t_array)): # percorre a matriz de t_array

        t = t_array[i]  # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        erro = erro_array[i]  # vai adotar o vetor de erro presente na matriz erro_array correspondente a posição i 
        color = colors[ i% len(colors)]
        plt.plot(t, erro, marker='o', linestyle='', color=color, label='Passo ' + str(h[i]))

    plt.title('Letra C - Erro Percentual Verdadeiro vs Tempo [%]')
    plt.xlabel('Tempo(s)')
    plt.ylabel('Erro Percentual Verdadeiro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0, "s"])
    tabela.add_row(["tf", tf, "s"])
    tabela.add_row(["y0", y0, "-"])
    tabela.add_row(["h", h, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable(['t(s)', 'y(t)', 'y_analitica(t)'])
    for i in range(len(t_array)):
        t = t_array[i]
        y = y_array[i]
        y_exata = y_exata_array[i]
        for val1, val2, val3 in zip(t, y, y_exata):
            tabela.add_row([val1, val2, val3])

    print(tabela)

letra_c = calculate_letra_c_rk2()
