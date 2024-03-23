#Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 
from prettytable import PrettyTable 

def calculate__comparacao():

    # Método de Euler-----------------------------------------------------------------------------
    # Definindo Variáveis de Entrada:
    t0_euler = 0
    tf_euler  = 2
    y0_euler  = 1

    h_euler  = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_euler  = len(h_euler ) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf_euler:float, t0_euler: float, h_euler:float) -> np.ndarray:

        n_euler  = np.zeros(len(h_euler ))
        for i in range(len(h_euler )):

            n_euler [i] = ((tf_euler -t0_euler )/(h_euler [i])) +1 

        return n_euler 
    
    n_euler  = calculate_n(tf_euler, t0_euler, h_euler )
    print(n_euler ) # vetor com os números de elementos de cada passo solicitado pelo problema 

    # Definindo a Função EDO:
    def f_euler(t_euler, y_euler):
        return y_euler*t_euler**3 - 1.5*y_euler 

    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_euler = []
    y_array_euler = []

    # Definindo as Condições Iniciais:
    for i in range(0,x_euler): # loop que vai de 0 ao tamanho do vetor de passos

        t_euler = np.linspace(t0_euler,tf_euler,int(n_euler[i])) # cria um vetor de t, de t0 a tf com o número de elementos da interação (número de elementos no vetor, na posição i)
        y_euler = np.zeros_like(t_euler) # cria um vetor igual ao vetor criado para o tempo 
        y_euler[0] = y0_euler # estabelece que na primeira posição do vetor de y, a condição inicial é y0
        y_array_euler.append(y_euler) # guarda os vetores de y dentro da matriz y_array
        t_array_euler.append(t_euler) # guarda os vetores de t dentro da matriz y_array

    print(t_array_euler)
    print(y_array_euler)

    # Método de Euler
    for i in range(x_euler): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        t_euler = t_array_euler[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_euler = y_array_euler[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        m_euler = int(n_euler[i]) - 1 # vai mensurar o número de elementos menos 1 

        for j in range(0,m_euler): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2
            y_euler[j+1] = y_euler[j] + h_euler[i]*f_euler(t_euler[j], y_euler[j]) # equacionamento do método de euler 
        y_array_euler[i] = y_euler # guarda o vetor y dentro matriz y_array

    print(y_array_euler)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(y_array_euler)): # vai percorrer a matriz de y 

        t_euler = t_array_euler[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_euler = y_array_euler[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_euler, y_euler, linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h_euler[i]))
    
    plt.title('Letra B - Solução Numérica por Euler - dy/dt = y*t³ - 1.5*y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Definindo a Função da Solução Analítica:
    def calculate_yexata_euler(t_euler:np.ndarray) -> np.ndarray:

        y_exata_list_euler = []

        for i in range(len(t_euler)):
            
            y_exata_euler = np.exp(((t_euler[i]**4)/4) - 1.5*t_euler[i])
            y_exata_list_euler.append(y_exata_euler)

        return y_exata_list_euler

    y_exata_array_euler = [] # cria uma matriz para inserir os vetores, com os valores de solução analítica, que vão ser gerados para cada vetor de t da matriz t_array

    for i in range(len(y_array_euler)): # percorre a matriz de soluções analíticas y_array

        t_euler = t_array_euler[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata_euler = calculate_yexata_euler(t_euler) # calcula a solução exata com a função da solução analítica 
        y_exata_array_euler.append(y_exata_euler) # guarda o vetor y_exata dentro da matriz y_exata_array
 
    print('y_exata_array', y_exata_array_euler)

    # Plotagem da Solução Analítica:
    colors = ['#006400', '#7CFC00']   # Dark green, medium spring green

    for i in range(len(t_array_euler)): # percorre a matriz de tempos t_array

        t_euler = t_array_euler[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata_euler = y_exata_array_euler[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i 
        color = colors[i % len(colors)]
        plt.plot(t_euler, y_exata_euler, linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h_euler[i]))

    plt.title('Letra B - Solução Analítica - y(t) = e(t^4/4 - 1.5*t)')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Comparação das Soluções Analíticas e Numéricas:
    colors_numerica = '#006400'
    colors_analitica = '#FF1493'

    for i in range(len(t_array_euler)): # percorre a matriz de tempos t_array

        t_euler = t_array_euler[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_euler = y_array_euler[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata_euler = y_exata_array_euler[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        plt.plot(t_euler, y_euler, linestyle='-', color=colors_numerica, label='Solução Numérica - Passo ' + str(h_euler[i]))
        plt.plot(t_euler, y_exata_euler, linestyle='-', color=colors_analitica, label='Solução Analítica - Passo ' + str(h_euler[i]))

    plt.title('Letra B - Comparação das Soluções Analítica e Numérica')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Cálculo do Erro Percentual Verdadeiro:
    def calculate_erro_euler(y_exata_euler:np.ndarray, y_euler:np.ndarray) -> np.ndarray: # a função devolve uma lista/vetor

        erro_list_euler = []
        for i in range(len(y_exata_euler)): # percorre o vetor de y_exata 
            
            erro_euler = np.abs((y_exata_euler[i]-y_euler[i])/y_exata_euler[i])*100
            erro_list_euler.append(erro_euler) # guarda o valor dentro do vetor erro_list
        
        return erro_list_euler

    erro_array_euler = [] # cria uma matriz para guardar os vetores de erro_list correspondente a cada vetor de y_exata presente na matriz y_exata_array

    for i in range(len(y_array_euler)): # percorre a matrzi de t_array 

        y_euler = y_array_euler[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata_euler = y_exata_array_euler[i]  # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        erro_euler = calculate_erro_euler(y_exata_euler, y_euler) # calcula o erro 
        erro_array_euler.append(erro_euler) # guarda o vetor de erros erro_list dentro da matriz erro_arrat

    print('erro', erro_array_euler)

    # Plotagem do Erro Percentual Verdadeiro vs Tempo:
    colors = ['#FF4500', '#FF8C00']   # Dark orange, orange

    for i in range(len(t_array_euler)): # percorre a matriz de t_array

        t_euler = t_array_euler[i]  # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        erro_euler = erro_array_euler[i]  # vai adotar o vetor de erro presente na matriz erro_array correspondente a posição i 
        color = colors[ i% len(colors)]
        plt.plot(t_euler, erro_euler, marker='o', linestyle='', color=color, label='Erro Percentual Verdadeiro [%] - Passo ' + str(h_euler[i]))

    plt.title('Letra B - Erro Percentual Verdadeiro vs Tempo')
    plt.xlabel('Tempo(s)')
    plt.ylabel('Erro Percentual Verdadeiro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_euler, "s"])
    tabela.add_row(["tf", tf_euler, "s"])
    tabela.add_row(["y0", y0_euler, "-"])
    tabela.add_row(["h", h_euler, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela_euler = PrettyTable(['t(s)', 'y(t)', 'y_analítica(t)'])

    for i in range(len(t_array_euler)):
        t_euler = t_array_euler[i]
        y_euler = y_array_euler[i]
        y_exata_euler = y_exata_array_euler[i]
        for val1, val2, val3 in zip(t_euler, y_euler, y_exata_euler):
            tabela.add_row([val1, val2, val3])

    print(tabela)
    
    # Método de RK2----------------------------------------------------------
    # Definindo Variáveis de Entrada:
    t0_rk2 = 0
    tf_rk2  = 2
    y0_rk2  = 1

    h_rk2  = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk2  = len(h_rk2 ) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf_rk2:float, t0_rk2: float, h_rk2:float) -> np.ndarray:

        n_rk2  = np.zeros(len(h_rk2 ))
        for i in range(len(h_rk2 )):

            n_rk2 [i] = ((tf_rk2 -t0_rk2 )/(h_rk2[i])) +1 

        return n_rk2 
    
    n_rk2  = calculate_n(tf_rk2, t0_rk2, h_rk2 )
    print(n_rk2 ) # vetor com os números de elementos de cada passo solicitado pelo problema 

    # Definindo a Função EDO:
    def f_rk2(t_rk2, y_rk2):
        return y_rk2*t_rk2**3 - 1.5*y_rk2 

    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk2 = []
    y_array_rk2 = []

    # Definindo as Condições Iniciais:
    for i in range(0,x_rk2): # loop que vai de 0 ao tamanho do vetor de passos

        t_rk2 = np.linspace(t0_rk2,tf_rk2,int(n_rk2[i])) # cria um vetor de t, de t0 a tf com o número de elementos da interação (número de elementos no vetor, na posição i)
        y_rk2 = np.zeros_like(t_rk2) # cria um vetor igual ao vetor criado para o tempo 
        y_rk2[0] = y0_rk2 # estabelece que na primeira posição do vetor de y, a condição inicial é y0
        y_array_rk2.append(y_rk2) # guarda os vetores de y dentro da matriz y_array
        t_array_rk2.append(t_rk2) # guarda os vetores de t dentro da matriz y_array

    print(t_array_rk2)
    print(y_array_rk2)

    # Método de rk2
    for i in range(x_rk2): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        t_rk2 = t_array_rk2[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_rk2 = y_array_rk2[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        m_rk2 = int(n_rk2[i]) - 1 # vai mensurar o número de elementos menos 1 

        for j in range(0,m_rk2): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2
            k1_rk2 = h_rk2[i]*f_rk2(t_rk2[j], y_rk2[j])
            k2_rk2 = h_rk2[i]*f_rk2(t_rk2[j+1], y_rk2[j] + k1_rk2)
            y_rk2[j+1] = y_rk2[j] + (1/2)*(k1_rk2 + k2_rk2)# método de runge-kutta 2ª ordem 
        y_array_rk2[i] = y_rk2 # guarda o vetor y dentro matriz y_array

    print(y_array_rk2)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(y_array_rk2)): # vai percorrer a matriz de y 

        t_rk2 = t_array_rk2[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_rk2 = y_array_rk2[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk2, y_rk2, linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h_rk2[i]))
    
    plt.title('Letra B - Solução Numérica por rk2 - dy/dt = y*t³ - 1.5*y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Definindo a Função da Solução Analítica:
    def calculate_yexata_rk2(t_rk2:np.ndarray) -> np.ndarray:

        y_exata_list_rk2 = []

        for i in range(len(t_rk2)):
            
            y_exata_rk2 = np.exp(((t_rk2[i]**4)/4) - 1.5*t_rk2[i])
            y_exata_list_rk2.append(y_exata_rk2)

        return y_exata_list_rk2

    y_exata_array_rk2 = [] # cria uma matriz para inserir os vetores, com os valores de solução analítica, que vão ser gerados para cada vetor de t da matriz t_array

    for i in range(len(y_array_rk2)): # percorre a matriz de soluções analíticas y_array

        t_rk2 = t_array_rk2[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata_rk2 = calculate_yexata_rk2(t_rk2) # calcula a solução exata com a função da solução analítica 
        y_exata_array_rk2.append(y_exata_rk2) # guarda o vetor y_exata dentro da matriz y_exata_array
 
    print('y_exata_array', y_exata_array_rk2)

    # Plotagem da Solução Analítica:
    colors = ['#006400', '#7CFC00']   # Dark green, medium spring green

    for i in range(len(t_array_rk2)): # percorre a matriz de tempos t_array

        t_rk2 = t_array_rk2[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata_rk2 = y_exata_array_rk2[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i 
        color = colors[i % len(colors)]
        plt.plot(t_rk2, y_exata_rk2, linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h_rk2[i]))

    plt.title('Letra B - Solução Analítica - y(t) = e(t^4/4 - 1.5*t)')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Comparação das Soluções Analíticas e Numéricas:
    colors_numerica = '#006400'
    colors_analitica = '#FF1493'

    for i in range(len(t_array_rk2)): # percorre a matriz de tempos t_array

        t_rk2 = t_array_rk2[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_rk2 = y_array_rk2[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata_rk2 = y_exata_array_rk2[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        plt.plot(t_rk2, y_rk2, linestyle='-', color=colors_numerica, label='Solução Numérica - Passo ' + str(h_rk2[i]))
        plt.plot(t_rk2, y_exata_rk2, linestyle='-', color=colors_analitica, label='Solução Analítica - Passo ' + str(h_rk2[i]))

    plt.title('Letra B - Comparação das Soluções Analítica e Numérica')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Cálculo do Erro Percentual Verdadeiro:
    def calculate_erro_rk2(y_exata_rk2:np.ndarray, y_rk2:np.ndarray) -> np.ndarray: # a função devolve uma lista/vetor

        erro_list_rk2 = []
        for i in range(len(y_exata_rk2)): # percorre o vetor de y_exata 
            
            erro_rk2 = np.abs((y_exata_rk2[i]-y_rk2[i])/y_exata_rk2[i])*100
            erro_list_rk2.append(erro_rk2) # guarda o valor dentro do vetor erro_list
        
        return erro_list_rk2

    erro_array_rk2 = [] # cria uma matriz para guardar os vetores de erro_list correspondente a cada vetor de y_exata presente na matriz y_exata_array

    for i in range(len(y_array_rk2)): # percorre a matrzi de t_array 

        y_rk2 = y_array_rk2[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata_rk2 = y_exata_array_rk2[i]  # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        erro_rk2 = calculate_erro_rk2(y_exata_rk2, y_rk2) # calcula o erro 
        erro_array_rk2.append(erro_rk2) # guarda o vetor de erros erro_list dentro da matriz erro_arrat

    print('erro', erro_array_rk2)

    # Plotagem do Erro Percentual Verdadeiro vs Tempo:
    colors = ['#FF4500', '#FF8C00']   # Dark orange, orange

    for i in range(len(t_array_rk2)): # percorre a matriz de t_array

        t_rk2 = t_array_rk2[i]  # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        erro_rk2 = erro_array_rk2[i]  # vai adotar o vetor de erro presente na matriz erro_array correspondente a posição i 
        color = colors[ i% len(colors)]
        plt.plot(t_rk2, erro_rk2, marker='o', linestyle='', color=color, label='Erro Percentual Verdadeiro [%] - Passo ' + str(h_rk2[i]))

    plt.title('Letra B - Erro Percentual Verdadeiro vs Tempo')
    plt.xlabel('Tempo(s)')
    plt.ylabel('Erro Percentual Verdadeiro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk2, "s"])
    tabela.add_row(["tf", tf_rk2, "s"])
    tabela.add_row(["y0", y0_rk2, "-"])
    tabela.add_row(["h", h_rk2, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela_rk2 = PrettyTable(['t(s)', 'y(t)', 'y_analítica(t)'])

    for i in range(len(t_array_rk2)):
        t_rk2 = t_array_rk2[i]
        y_rk2 = y_array_rk2[i]
        y_exata_rk2 = y_exata_array_rk2[i]
        for val1, val2, val3 in zip(t_rk2, y_rk2, y_exata_rk2):
            tabela.add_row([val1, val2, val3])

    print(tabela)

    # Método de RK4----------------------------------------------------------
    # Definindo Variáveis de Entrada:
    t0_rk4 = 0
    tf_rk4  = 2
    y0_rk4  = 1

    h_rk4  = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk4  = len(h_rk4 ) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf_rk4:float, t0_rk4: float, h_rk4:float) -> np.ndarray:

        n_rk4  = np.zeros(len(h_rk4 ))
        for i in range(len(h_rk4 )):

            n_rk4 [i] = ((tf_rk4 -t0_rk4 )/(h_rk4[i])) +1 

        return n_rk4 
    
    n_rk4  = calculate_n(tf_rk4, t0_rk4, h_rk4 )
    print(n_rk4 ) # vetor com os números de elementos de cada passo solicitado pelo problema 

    # Definindo a Função EDO:
    def f_rk4(t_rk4, y_rk4):
        return y_rk4*t_rk4**3 - 1.5*y_rk4 

    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk4 = []
    y_array_rk4 = []

    # Definindo as Condições Iniciais:
    for i in range(0,x_rk4): # loop que vai de 0 ao tamanho do vetor de passos

        t_rk4 = np.linspace(t0_rk4,tf_rk4,int(n_rk4[i])) # cria um vetor de t, de t0 a tf com o número de elementos da interação (número de elementos no vetor, na posição i)
        y_rk4 = np.zeros_like(t_rk4) # cria um vetor igual ao vetor criado para o tempo 
        y_rk4[0] = y0_rk4 # estabelece que na primeira posição do vetor de y, a condição inicial é y0
        y_array_rk4.append(y_rk4) # guarda os vetores de y dentro da matriz y_array
        t_array_rk4.append(t_rk4) # guarda os vetores de t dentro da matriz y_array

    print(t_array_rk4)
    print(y_array_rk4)

    # Método de rk4
    for i in range(x_rk4): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        t_rk4 = t_array_rk4[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_rk4 = y_array_rk4[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        m_rk4 = int(n_rk4[i]) - 1 # vai mensurar o número de elementos menos 1 

        for j in range(0,m_rk4): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 1s (y[i+1] é 2), sendo que o tempo final é 2
            k1_rk4 = h_rk4[i]*f_rk4(t_rk4[j], y_rk4[j])
            k2_rk4 = h_rk4[i]*f_rk4(t_rk4 [j] + (h_rk4[i]/2), y_rk4[j] + (k1_rk4/2))
            k3_rk4 = h_rk4[i]*f_rk4(t_rk4[j] + (h_rk4[i]/2), y_rk4[j] + (k2_rk4/2))
            k4_rk4 = h_rk4[i]*f_rk4(t_rk4[j] + h_rk4[i], y_rk4[j] + k3_rk4)
            y_rk4[j+1] = y_rk4[j] + (1/6)*(k1_rk4 + (2*k2_rk4) + (2*k3_rk4) + k4_rk4) # método de runge-kutta de 4ª ordem
        y_array_rk4[i] = y_rk4 # guarda o vetor y dentro matriz y_array

    print(y_array_rk4)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(y_array_rk4)): # vai percorrer a matriz de y 

        t_rk4 = t_array_rk4[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_rk4 = y_array_rk4[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk4, y_rk4, linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h_rk4[i]))
    
    plt.title('Letra B - Solução Numérica por rk4 - dy/dt = y*t³ - 1.5*y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Definindo a Função da Solução Analítica:
    def calculate_yexata_rk4(t_rk4:np.ndarray) -> np.ndarray:

        y_exata_list_rk4 = []

        for i in range(len(t_rk4)):
            
            y_exata_rk4 = np.exp(((t_rk4[i]**4)/4) - 1.5*t_rk4[i])
            y_exata_list_rk4.append(y_exata_rk4)

        return y_exata_list_rk4

    y_exata_array_rk4 = [] # cria uma matriz para inserir os vetores, com os valores de solução analítica, que vão ser gerados para cada vetor de t da matriz t_array

    for i in range(len(y_array_rk4)): # percorre a matriz de soluções analíticas y_array

        t_rk4 = t_array_rk4[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata_rk4 = calculate_yexata_rk4(t_rk4) # calcula a solução exata com a função da solução analítica 
        y_exata_array_rk4.append(y_exata_rk4) # guarda o vetor y_exata dentro da matriz y_exata_array
 
    print('y_exata_array', y_exata_array_rk4)

    # Plotagem da Solução Analítica:
    colors = ['#006400', '#7CFC00']   # Dark green, medium spring green

    for i in range(len(t_array_rk4)): # percorre a matriz de tempos t_array

        t_rk4 = t_array_rk4[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_exata_rk4 = y_exata_array_rk4[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i 
        color = colors[i % len(colors)]
        plt.plot(t_rk4, y_exata_rk4, linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h_rk4[i]))

    plt.title('Letra B - Solução Analítica - y(t) = e(t^4/4 - 1.5*t)')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Comparação das Soluções Analíticas e Numéricas:
    colors_numerica = '#006400'
    colors_analitica = '#FF1493'

    for i in range(len(t_array_rk4)): # percorre a matriz de tempos t_array

        t_rk4 = t_array_rk4[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        y_rk4 = y_array_rk4[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata_rk4 = y_exata_array_rk4[i] # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        plt.plot(t_rk4, y_rk4, linestyle='-', color=colors_numerica, label='Solução Numérica - Passo ' + str(h_rk4[i]))
        plt.plot(t_rk4, y_exata_rk4, linestyle='-', color=colors_analitica, label='Solução Analítica - Passo ' + str(h_rk4[i]))

    plt.title('Letra B - Comparação das Soluções Analítica e Numérica')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Cálculo do Erro Percentual Verdadeiro:
    def calculate_erro_rk4(y_exata_rk4:np.ndarray, y_rk4:np.ndarray) -> np.ndarray: # a função devolve uma lista/vetor

        erro_list_rk4 = []
        for i in range(len(y_exata_rk4)): # percorre o vetor de y_exata 
            
            erro_rk4 = np.abs((y_exata_rk4[i]-y_rk4[i])/y_exata_rk4[i])*100
            erro_list_rk4.append(erro_rk4) # guarda o valor dentro do vetor erro_list
        
        return erro_list_rk4

    erro_array_rk4 = [] # cria uma matriz para guardar os vetores de erro_list correspondente a cada vetor de y_exata presente na matriz y_exata_array

    for i in range(len(y_array_rk4)): # percorre a matrzi de t_array 

        y_rk4 = y_array_rk4[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        y_exata_rk4 = y_exata_array_rk4[i]  # vai adotar o vetor de y_exata pesente na matriz y_exata_array correspondente a posição i
        erro_rk4 = calculate_erro_rk4(y_exata_rk4, y_rk4) # calcula o erro 
        erro_array_rk4.append(erro_rk4) # guarda o vetor de erros erro_list dentro da matriz erro_arrat

    print('erro', erro_array_rk4)

    # Plotagem do Erro Percentual Verdadeiro vs Tempo:
    colors = ['#FF4500', '#FF8C00']   # Dark orange, orange

    for i in range(len(t_array_rk4)): # percorre a matriz de t_array

        t_rk4 = t_array_rk4[i]  # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        erro_rk4 = erro_array_rk4[i]  # vai adotar o vetor de erro presente na matriz erro_array correspondente a posição i 
        color = colors[ i% len(colors)]
        plt.plot(t_rk4, erro_rk4, marker='o', linestyle='', color=color, label='Erro Percentual Verdadeiro [%] - Passo ' + str(h_rk4[i]))

    plt.title('Letra B - Erro Percentual Verdadeiro vs Tempo')
    plt.xlabel('Tempo(s)')
    plt.ylabel('Erro Percentual Verdadeiro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk4, "s"])
    tabela.add_row(["tf", tf_rk4, "s"])
    tabela.add_row(["y0", y0_rk4, "-"])
    tabela.add_row(["h", h_rk4, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela_rk4 = PrettyTable(['t(s)', 'y(t)', 'y_analítica(t)'])

    for i in range(len(t_array_rk4)):
        t_rk4 = t_array_rk4[i]
        y_rk4 = y_array_rk4[i]
        y_exata_rk4 = y_exata_array_rk4[i]
        for val1, val2, val3 in zip(t_rk4, y_rk4, y_exata_rk4):
            tabela.add_row([val1, val2, val3])

    print(tabela)

    # Plotagem da Comparação das Soluções Numéricas:
    for i in range(len(t_array_euler)):

        t = t_array_euler[i]
        y_euler = y_array_euler[i]
        y_rk2 = y_array_rk2[i]
        y_rk4 = y_array_rk4[i]
        plt.plot(t, y_euler, linestyle='-', color='#FF1493', label='Solução Numérica - Euler - Passo ' + str(h_euler[i]))
        plt.plot(t, y_rk2, linestyle='-', color='#4B0082', label='Solução Numérica - RK2 - Passo ' + str(h_rk2[i]))
        plt.plot(t, y_rk4, linestyle='-', color='#006400', label='Solução Numérica - RK4 - Passo ' + str(h_rk4[i]))

    plt.title('Comparação das Soluções Numéricas')
    plt.xlabel('Tempo(s)')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotagem da Comparação dos Erros Percentuais Verdadeiros:
    for i in range(len(t_array_euler)):

        t = t_array_euler[i]
        erro_euler = erro_array_euler[i]
        erro_rk2 = erro_array_rk2[i]
        erro_rk4 = erro_array_rk4[i]
        plt.plot(t, erro_euler, linestyle='-', color='#FF1493', label='Erro Percentual Verdadeiro [%] - Euler - Passo ' + str(h_euler[i]))
        plt.plot(t, erro_rk2, linestyle='-', color='#4B0082', label='Erro Percentual Verdadeiro [%] - Euler - Passo ' + str(h_rk2[i]))
        plt.plot(t, erro_rk4, linestyle='-', color='#006400', label='Erro Percentual Verdadeiro [%] - Euler - Passo ' + str(h_rk4[i]))

    plt.title('Comparação dos Erros Percentuais Verdadeiros')
    plt.xlabel('Tempo(s)')
    plt.ylabel('Erro Percentual Verdadeiro [%]')
    plt.grid(True)
    plt.legend()
    plt.show()

comp = calculate__comparacao()