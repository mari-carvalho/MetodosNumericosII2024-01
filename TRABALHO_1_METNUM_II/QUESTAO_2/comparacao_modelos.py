# Questão 2 - Comparação dos Modelos

# Resolução do Problema de PVI:
# Importando Bibliotecas =:
import numpy as np
import math as mt 
import matplotlib.pyplot as plt
from prettytable import PrettyTable 

def calculate_comparacao_modelos():
    
    # Método de euler_aula ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_euler_aula = 9.81 # m/s²
    d_orificio_euler_aula = 0.03 # m 
    r_euler_aula = 1.5
    h0_euler_aula = 2.75 # m 
    C_euler_aula = 0.55 
    t0_euler_aula = 0 
    tf_euler_aula = 1000
    
    p_euler_aula = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_euler_aula = len(p_euler_aula) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_euler_aula(tf_euler_aula:float, t0_euler_aula: float, p_euler_aula:float) -> np.ndarray:

        n_euler_aula = np.zeros(len(p_euler_aula))
        for i in range(len(p_euler_aula)):

            n_euler_aula[i] = ((tf_euler_aula-t0_euler_aula)/(p_euler_aula[i])) +1 

        return n_euler_aula
    
    n_euler_aula = calculate_n_euler_aula(tf_euler_aula,t0_euler_aula,p_euler_aula)
    print('n', n_euler_aula) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_euler_aula(d_orificio_euler_aula:float) -> float: 
        
        area_euler_aula = (np.pi)*(d_orificio_euler_aula/2)**2

        return area_euler_aula
    area_euler_aula = calculate_area_euler_aula(d_orificio_euler_aula)   
    print(area_euler_aula)  

    def calculate_Q_euler_aula(C_euler_aula:float, area_euler_aula:float, g_euler_aula:float, h0_euler_aula:float) -> float:
        
        Q_euler_aula = C_euler_aula*area_euler_aula*np.sqrt(2*g_euler_aula*h0_euler_aula)

        return Q_euler_aula
    
    Q_euler_aula = calculate_Q_euler_aula(C_euler_aula, area_euler_aula, g_euler_aula, h0_euler_aula)
    print(Q_euler_aula)

    def calculate_V_euler_aula(r_euler_aula:float) -> float:

        V_euler_aula = 4/3 * np.pi * r_euler_aula**3

        return V_euler_aula
    
    V_euler_aula = calculate_V_euler_aula(r_euler_aula)
    print(V_euler_aula)

    def calculate_t_estimativa_euler_aula(V_euler_aula:float, Q_euler_aula:float) -> float:
    
        t_estimativa_euler_aula = V_euler_aula/Q_euler_aula

        return t_estimativa_euler_aula
    
    t_estimativa_euler_aula = calculate_t_estimativa_euler_aula(V_euler_aula, Q_euler_aula)
    print('t_estimativa', t_estimativa_euler_aula)

    # Definindo a Função EDO:
    def f_euler_aula(t_euler_aula,h_euler_aula):
        return (-C_euler_aula * area_euler_aula * np.sqrt(2*g_euler_aula))/(np.sqrt(h_euler_aula)*((2*np.pi*r_euler_aula) - (np.pi*h_euler_aula)))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_euler_aula = []
    h_array_euler_aula = []

    # Método de euler_aula
    for i in range(x_euler_aula): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_euler_aula = []
        t_list_euler_aula = []
        m_euler_aula = int(n_euler_aula[i]) - 1 
        j_euler_aula = 0
        h_euler_aula = 2.75
        t_euler_aula = 0
        while h_euler_aula > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_euler_aula = h_euler_aula + p_euler_aula[i]*f_euler_aula(t_euler_aula, h_euler_aula)
                if h_euler_aula <= 0:
                    break
                else:
                    t_euler_aula = t_euler_aula + p_euler_aula[i]
                    t_list_euler_aula.append(t_euler_aula)
                    h_list_euler_aula.append(h_euler_aula)
        t_array_euler_aula.append(t_list_euler_aula)
        h_array_euler_aula.append(h_list_euler_aula)# guarda o vetor y dentro matriz y_array

    print('h', h_array_euler_aula)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_euler_aula)): # vai percorrer a matriz de y 

        t_euler_aula = t_array_euler_aula[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_euler_aula = h_array_euler_aula[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_euler_aula, h_euler_aula, linestyle='-', color=color, label='Passo ' + str(p_euler_aula[i]))
    
    plt.title('Questão 2 - Solução Numérica por Euler-Aula')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_euler_aula, "s"])
    tabela.add_row(["tf", tf_euler_aula, "s"])
    tabela.add_row(["h0", h0_euler_aula, "m"])
    tabela.add_row(["r", r_euler_aula, "m"])
    tabela.add_row(["d_orifício", d_orificio_euler_aula, "m"])
    tabela.add_row(["g", g_euler_aula, "m/s²"])
    tabela.add_row(["C", C_euler_aula, "-"])
    tabela.add_row(["p", p_euler_aula, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_euler_aula, "m³/s"])
    tabela.add_row(["V", V_euler_aula, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_euler_aula, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_euler_aula)):
        t_euler_aula = t_array_euler_aula[i]
        h_euler_aula = h_array_euler_aula[i]
        for val1, val2 in zip(t_euler_aula, h_euler_aula):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de rk2_aula ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_rk2_aula = 9.81 # m/s²
    d_orificio_rk2_aula = 0.03 # m 
    r_rk2_aula = 1.5
    h0_rk2_aula = 2.75 # m 
    C_rk2_aula = 0.55 
    t0_rk2_aula = 0 
    tf_rk2_aula = 1000
    
    p_rk2_aula = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk2_aula = len(p_rk2_aula) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_rk2_aula(tf_rk2_aula:float, t0_rk2_aula: float, p_rk2_aula:float) -> np.ndarray:

        n_rk2_aula = np.zeros(len(p_rk2_aula))
        for i in range(len(p_rk2_aula)):

            n_rk2_aula[i] = ((tf_rk2_aula-t0_rk2_aula)/(p_rk2_aula[i])) +1 

        return n_rk2_aula
    
    n_rk2_aula = calculate_n_rk2_aula(tf_rk2_aula,t0_rk2_aula,p_rk2_aula)
    print('n', n_rk2_aula) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_rk2_aula(d_orificio_rk2_aula:float) -> float: 
        
        area_rk2_aula = (np.pi)*(d_orificio_rk2_aula/2)**2

        return area_rk2_aula
    area_rk2_aula = calculate_area_rk2_aula(d_orificio_rk2_aula)   
    print(area_rk2_aula)  

    def calculate_Q_rk2_aula(C_rk2_aula:float, area_rk2_aula:float, g_rk2_aula:float, h0_rk2_aula:float) -> float:
        
        Q_rk2_aula = C_rk2_aula*area_rk2_aula*np.sqrt(2*g_rk2_aula*h0_rk2_aula)

        return Q_rk2_aula
    
    Q_rk2_aula = calculate_Q_rk2_aula(C_rk2_aula, area_rk2_aula, g_rk2_aula, h0_rk2_aula)
    print(Q_rk2_aula)

    def calculate_V_rk2_aula(r_rk2_aula:float) -> float:

        V_rk2_aula = 4/3 * np.pi * r_rk2_aula**3

        return V_rk2_aula
    
    V_rk2_aula = calculate_V_rk2_aula(r_rk2_aula)
    print(V_rk2_aula)

    def calculate_t_estimativa_rk2_aula(V_rk2_aula:float, Q_rk2_aula:float) -> float:
    
        t_estimativa_rk2_aula = V_rk2_aula/Q_rk2_aula

        return t_estimativa_rk2_aula
    
    t_estimativa_rk2_aula = calculate_t_estimativa_rk2_aula(V_rk2_aula, Q_rk2_aula)
    print('t_estimativa', t_estimativa_rk2_aula)

    # Definindo a Função EDO:
    def f_rk2_aula(t_rk2_aula,h_rk2_aula):
        return (-C_rk2_aula * area_rk2_aula * np.sqrt(2*g_rk2_aula))/(np.sqrt(h_rk2_aula)*((2*np.pi*r_rk2_aula) - (np.pi*h_rk2_aula)))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk2_aula = []
    h_array_rk2_aula = []

    # Método de rk2_aula
    for i in range(x_rk2_aula): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_rk2_aula = []
        t_list_rk2_aula = []
        m_rk2_aula = int(n_rk2_aula[i]) - 1 
        j_rk2_aula = 0
        h_rk2_aula = 2.75
        t_rk2_aula = 0
        while h_rk2_aula > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_rk2_aula = h_rk2_aula + p_rk2_aula[i]*f_rk2_aula(t_rk2_aula, h_rk2_aula)
                if h_rk2_aula <= 0:
                    break
                else:
                    t_rk2_aula = t_rk2_aula + p_rk2_aula[i]
                    t_list_rk2_aula.append(t_rk2_aula)
                    h_list_rk2_aula.append(h_rk2_aula)
        t_array_rk2_aula.append(t_list_rk2_aula)
        h_array_rk2_aula.append(h_list_rk2_aula)# guarda o vetor y dentro matriz y_array

    print('h', h_array_rk2_aula)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_rk2_aula)): # vai percorrer a matriz de y 

        t_rk2_aula = t_array_rk2_aula[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_rk2_aula = h_array_rk2_aula[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk2_aula, h_rk2_aula, linestyle='-', color=color, label='Passo ' + str(p_rk2_aula[i]))
    
    plt.title('Questão 2 - Solução Numérica por RK2-Aula')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk2_aula, "s"])
    tabela.add_row(["tf", tf_rk2_aula, "s"])
    tabela.add_row(["h0", h0_rk2_aula, "m"])
    tabela.add_row(["r", r_rk2_aula, "m"])
    tabela.add_row(["d_orifício", d_orificio_rk2_aula, "m"])
    tabela.add_row(["g", g_rk2_aula, "m/s²"])
    tabela.add_row(["C", C_rk2_aula, "-"])
    tabela.add_row(["p", p_rk2_aula, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_rk2_aula, "m³/s"])
    tabela.add_row(["V", V_rk2_aula, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_rk2_aula, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_rk2_aula)):
        t_rk2_aula = t_array_rk2_aula[i]
        h_rk2_aula = h_array_rk2_aula[i]
        for val1, val2 in zip(t_rk2_aula, h_rk2_aula):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de rk4_aula ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_rk4_aula = 9.81 # m/s²
    d_orificio_rk4_aula = 0.03 # m 
    r_rk4_aula = 1.5
    h0_rk4_aula = 2.75 # m 
    C_rk4_aula = 0.55 
    t0_rk4_aula = 0 
    tf_rk4_aula = 1000
    
    p_rk4_aula = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk4_aula = len(p_rk4_aula) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_rk4_aula(tf_rk4_aula:float, t0_rk4_aula: float, p_rk4_aula:float) -> np.ndarray:

        n_rk4_aula = np.zeros(len(p_rk4_aula))
        for i in range(len(p_rk4_aula)):

            n_rk4_aula[i] = ((tf_rk4_aula-t0_rk4_aula)/(p_rk4_aula[i])) +1 

        return n_rk4_aula
    
    n_rk4_aula = calculate_n_rk4_aula(tf_rk4_aula,t0_rk4_aula,p_rk4_aula)
    print('n', n_rk4_aula) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_rk4_aula(d_orificio_rk4_aula:float) -> float: 
        
        area_rk4_aula = (np.pi)*(d_orificio_rk4_aula/2)**2

        return area_rk4_aula
    area_rk4_aula = calculate_area_rk4_aula(d_orificio_rk4_aula)   
    print(area_rk4_aula)  

    def calculate_Q_rk4_aula(C_rk4_aula:float, area_rk4_aula:float, g_rk4_aula:float, h0_rk4_aula:float) -> float:
        
        Q_rk4_aula = C_rk4_aula*area_rk4_aula*np.sqrt(2*g_rk4_aula*h0_rk4_aula)

        return Q_rk4_aula
    
    Q_rk4_aula = calculate_Q_rk4_aula(C_rk4_aula, area_rk4_aula, g_rk4_aula, h0_rk4_aula)
    print(Q_rk4_aula)

    def calculate_V_rk4_aula(r_rk4_aula:float) -> float:

        V_rk4_aula = 4/3 * np.pi * r_rk4_aula**3

        return V_rk4_aula
    
    V_rk4_aula = calculate_V_rk4_aula(r_rk4_aula)
    print(V_rk4_aula)

    def calculate_t_estimativa_rk4_aula(V_rk4_aula:float, Q_rk4_aula:float) -> float:
    
        t_estimativa_rk4_aula = V_rk4_aula/Q_rk4_aula

        return t_estimativa_rk4_aula
    
    t_estimativa_rk4_aula = calculate_t_estimativa_rk4_aula(V_rk4_aula, Q_rk4_aula)
    print('t_estimativa', t_estimativa_rk4_aula)

    # Definindo a Função EDO:
    def f_rk4_aula(t_rk4_aula,h_rk4_aula):
        return (-C_rk4_aula * area_rk4_aula * np.sqrt(2*g_rk4_aula))/(np.sqrt(h_rk4_aula)*((2*np.pi*r_rk4_aula) - (np.pi*h_rk4_aula)))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk4_aula = []
    h_array_rk4_aula = []

    # Método de rk4_aula
    for i in range(x_rk4_aula): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_rk4_aula = []
        t_list_rk4_aula = []
        m_rk4_aula = int(n_rk4_aula[i]) - 1 
        j_rk4_aula = 0
        h_rk4_aula = 2.75
        t_rk4_aula = 0
        while h_rk4_aula > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_rk4_aula = h_rk4_aula + p_rk4_aula[i]*f_rk4_aula(t_rk4_aula, h_rk4_aula)
                if h_rk4_aula <= 0:
                    break
                else:
                    t_rk4_aula = t_rk4_aula + p_rk4_aula[i]
                    t_list_rk4_aula.append(t_rk4_aula)
                    h_list_rk4_aula.append(h_rk4_aula)
        t_array_rk4_aula.append(t_list_rk4_aula)
        h_array_rk4_aula.append(h_list_rk4_aula)# guarda o vetor y dentro matriz y_array

    print('h', h_array_rk4_aula)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_rk4_aula)): # vai percorrer a matriz de y 

        t_rk4_aula = t_array_rk4_aula[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_rk4_aula = h_array_rk4_aula[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk4_aula, h_rk4_aula, linestyle='-', color=color, label='Passo ' + str(p_rk4_aula[i]))
    
    plt.title('Questão 2 - Solução Numérica por RK4-Aula')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk4_aula, "s"])
    tabela.add_row(["tf", tf_rk4_aula, "s"])
    tabela.add_row(["h0", h0_rk4_aula, "m"])
    tabela.add_row(["r", r_rk4_aula, "m"])
    tabela.add_row(["d_orifício", d_orificio_rk4_aula, "m"])
    tabela.add_row(["g", g_rk4_aula, "m/s²"])
    tabela.add_row(["C", C_rk4_aula, "-"])
    tabela.add_row(["p", p_rk4_aula, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_rk4_aula, "m³/s"])
    tabela.add_row(["V", V_rk4_aula, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_rk4_aula, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_rk4_aula)):
        t_rk4_aula = t_array_rk4_aula[i]
        h_rk4_aula = h_array_rk4_aula[i]
        for val1, val2 in zip(t_rk4_aula, h_rk4_aula):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de euler_profe ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_euler_profe = 9.81 # m/s²
    d_orificio_euler_profe = 0.03 # m 
    r_euler_profe = 1.5
    h0_euler_profe = 2.75 # m 
    C_euler_profe = 0.55 
    t0_euler_profe = 0 
    tf_euler_profe = 1000
    
    p_euler_profe = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_euler_profe = len(p_euler_profe) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_euler_profe(tf_euler_profe:float, t0_euler_profe: float, p_euler_profe:float) -> np.ndarray:

        n_euler_profe = np.zeros(len(p_euler_profe))
        for i in range(len(p_euler_profe)):

            n_euler_profe[i] = ((tf_euler_profe-t0_euler_profe)/(p_euler_profe[i])) +1 

        return n_euler_profe
    
    n_euler_profe = calculate_n_euler_profe(tf_euler_profe,t0_euler_profe,p_euler_profe)
    print('n', n_euler_profe) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_euler_profe(d_orificio_euler_profe:float) -> float: 
        
        area_euler_profe = (np.pi)*(d_orificio_euler_profe/2)**2

        return area_euler_profe
    area_euler_profe = calculate_area_euler_profe(d_orificio_euler_profe)   
    print(area_euler_profe)  

    def calculate_Q_euler_profe(C_euler_profe:float, area_euler_profe:float, g_euler_profe:float, h0_euler_profe:float) -> float:
        
        Q_euler_profe = C_euler_profe*area_euler_profe*np.sqrt(2*g_euler_profe*h0_euler_profe)

        return Q_euler_profe
    
    Q_euler_profe = calculate_Q_euler_profe(C_euler_profe, area_euler_profe, g_euler_profe, h0_euler_profe)
    print(Q_euler_profe)

    def calculate_V_euler_profe(r_euler_profe:float) -> float:

        V_euler_profe = 4/3 * np.pi * r_euler_profe**3

        return V_euler_profe
    
    V_euler_profe = calculate_V_euler_profe(r_euler_profe)
    print(V_euler_profe)

    def calculate_t_estimativa_euler_profe(V_euler_profe:float, Q_euler_profe:float) -> float:
    
        t_estimativa_euler_profe = V_euler_profe/Q_euler_profe

        return t_estimativa_euler_profe
    
    t_estimativa_euler_profe = calculate_t_estimativa_euler_profe(V_euler_profe, Q_euler_profe)
    print('t_estimativa', t_estimativa_euler_profe)

    # Definindo a Função EDO:
    def f_euler_profe(t_euler_profe,h_euler_profe):
        return (-C_euler_profe * area_euler_profe * np.sqrt(2*g_euler_profe*h_euler_profe))/((((2/3)*np.pi*h_euler_profe)*(3*r_euler_profe - h_euler_profe)) - ((np.pi*h_euler_profe**2)/3))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_euler_profe = []
    h_array_euler_profe = []

    # Método de euler_profe
    for i in range(x_euler_profe): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_euler_profe = []
        t_list_euler_profe = []
        m_euler_profe = int(n_euler_profe[i]) - 1 
        j_euler_profe = 0
        h_euler_profe = 2.75
        t_euler_profe = 0
        while h_euler_profe > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_euler_profe = h_euler_profe + p_euler_profe[i]*f_euler_profe(t_euler_profe, h_euler_profe)
                if h_euler_profe <= 0:
                    break
                else:
                    t_euler_profe = t_euler_profe + p_euler_profe[i]
                    t_list_euler_profe.append(t_euler_profe)
                    h_list_euler_profe.append(h_euler_profe)
        t_array_euler_profe.append(t_list_euler_profe)
        h_array_euler_profe.append(h_list_euler_profe)# guarda o vetor y dentro matriz y_array

    print('h', h_array_euler_profe)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_euler_profe)): # vai percorrer a matriz de y 

        t_euler_profe = t_array_euler_profe[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_euler_profe = h_array_euler_profe[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_euler_profe, h_euler_profe, linestyle='-', color=color, label='Passo ' + str(p_euler_profe[i]))
    
    plt.title('Questão 2 - Solução Numérica por Euler-Profe')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_euler_profe, "s"])
    tabela.add_row(["tf", tf_euler_profe, "s"])
    tabela.add_row(["h0", h0_euler_profe, "m"])
    tabela.add_row(["r", r_euler_profe, "m"])
    tabela.add_row(["d_orifício", d_orificio_euler_profe, "m"])
    tabela.add_row(["g", g_euler_profe, "m/s²"])
    tabela.add_row(["C", C_euler_profe, "-"])
    tabela.add_row(["p", p_euler_profe, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_euler_profe, "m³/s"])
    tabela.add_row(["V", V_euler_profe, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_euler_profe, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_euler_profe)):
        t_euler_profe = t_array_euler_profe[i]
        h_euler_profe = h_array_euler_profe[i]
        for val1, val2 in zip(t_euler_profe, h_euler_profe):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de rk2_profe ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_rk2_profe = 9.81 # m/s²
    d_orificio_rk2_profe = 0.03 # m 
    r_rk2_profe = 1.5
    h0_rk2_profe = 2.75 # m 
    C_rk2_profe = 0.55 
    t0_rk2_profe = 0 
    tf_rk2_profe = 1000
    
    p_rk2_profe = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk2_profe = len(p_rk2_profe) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_rk2_profe(tf_rk2_profe:float, t0_rk2_profe: float, p_rk2_profe:float) -> np.ndarray:

        n_rk2_profe = np.zeros(len(p_rk2_profe))
        for i in range(len(p_rk2_profe)):

            n_rk2_profe[i] = ((tf_rk2_profe-t0_rk2_profe)/(p_rk2_profe[i])) +1 

        return n_rk2_profe
    
    n_rk2_profe = calculate_n_rk2_profe(tf_rk2_profe,t0_rk2_profe,p_rk2_profe)
    print('n', n_rk2_profe) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_rk2_profe(d_orificio_rk2_profe:float) -> float: 
        
        area_rk2_profe = (np.pi)*(d_orificio_rk2_profe/2)**2

        return area_rk2_profe
    area_rk2_profe = calculate_area_rk2_profe(d_orificio_rk2_profe)   
    print(area_rk2_profe)  

    def calculate_Q_rk2_profe(C_rk2_profe:float, area_rk2_profe:float, g_rk2_profe:float, h0_rk2_profe:float) -> float:
        
        Q_rk2_profe = C_rk2_profe*area_rk2_profe*np.sqrt(2*g_rk2_profe*h0_rk2_profe)

        return Q_rk2_profe
    
    Q_rk2_profe = calculate_Q_rk2_profe(C_rk2_profe, area_rk2_profe, g_rk2_profe, h0_rk2_profe)
    print(Q_rk2_profe)

    def calculate_V_rk2_profe(r_rk2_profe:float) -> float:

        V_rk2_profe = 4/3 * np.pi * r_rk2_profe**3

        return V_rk2_profe
    
    V_rk2_profe = calculate_V_rk2_profe(r_rk2_profe)
    print(V_rk2_profe)

    def calculate_t_estimativa_rk2_profe(V_rk2_profe:float, Q_rk2_profe:float) -> float:
    
        t_estimativa_rk2_profe = V_rk2_profe/Q_rk2_profe

        return t_estimativa_rk2_profe
    
    t_estimativa_rk2_profe = calculate_t_estimativa_rk2_profe(V_rk2_profe, Q_rk2_profe)
    print('t_estimativa', t_estimativa_rk2_profe)

    # Definindo a Função EDO:
    def f_rk2_profe(t_rk2_profe,h_rk2_profe):
        return (-C_rk2_profe * area_rk2_profe  * np.sqrt(2*g_rk2_profe *h_rk2_profe ))/((((2/3)*np.pi*h_rk2_profe )*(3*r_rk2_profe  - h_rk2_profe )) - ((np.pi*h_rk2_profe **2)/3))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk2_profe = []
    h_array_rk2_profe = []

    # Método de rk2_profe
    for i in range(x_rk2_profe): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_rk2_profe = []
        t_list_rk2_profe = []
        m_rk2_profe = int(n_rk2_profe[i]) - 1 
        j_rk2_profe = 0
        h_rk2_profe = 2.75
        t_rk2_profe = 0
        while h_rk2_profe > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_rk2_profe = h_rk2_profe + p_rk2_profe[i]*f_rk2_profe(t_rk2_profe, h_rk2_profe)
                if h_rk2_profe <= 0:
                    break
                else:
                    t_rk2_profe = t_rk2_profe + p_rk2_profe[i]
                    t_list_rk2_profe.append(t_rk2_profe)
                    h_list_rk2_profe.append(h_rk2_profe)
        t_array_rk2_profe.append(t_list_rk2_profe)
        h_array_rk2_profe.append(h_list_rk2_profe)# guarda o vetor y dentro matriz y_array

    print('h', h_array_rk2_profe)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_rk2_profe)): # vai percorrer a matriz de y 

        t_rk2_profe = t_array_rk2_profe[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_rk2_profe = h_array_rk2_profe[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk2_profe, h_rk2_profe, linestyle='-', color=color, label='Solução Numérica- Passo ' + str(p_rk2_profe[i]))
    
    plt.title('Questão 2 - Solução Numérica por RK2-Profe')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk2_profe, "s"])
    tabela.add_row(["tf", tf_rk2_profe, "s"])
    tabela.add_row(["h0", h0_rk2_profe, "m"])
    tabela.add_row(["r", r_rk2_profe, "m"])
    tabela.add_row(["d_orifício", d_orificio_rk2_profe, "m"])
    tabela.add_row(["g", g_rk2_profe, "m/s²"])
    tabela.add_row(["C", C_rk2_profe, "-"])
    tabela.add_row(["p", p_rk2_profe, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_rk2_profe, "m³/s"])
    tabela.add_row(["V", V_rk2_profe, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_rk2_profe, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_rk2_profe)):
        t_rk2_profe = t_array_rk2_profe[i]
        h_rk2_profe = h_array_rk2_profe[i]
        for val1, val2 in zip(t_rk2_profe, h_rk2_profe):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de rk4_profe ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_rk4_profe = 9.81 # m/s²
    d_orificio_rk4_profe = 0.03 # m 
    r_rk4_profe = 1.5
    h0_rk4_profe = 2.75 # m 
    C_rk4_profe = 0.55 
    t0_rk4_profe = 0 
    tf_rk4_profe = 1000
    
    p_rk4_profe = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk4_profe = len(p_rk4_profe) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_rk4_profe(tf_rk4_profe:float, t0_rk4_profe: float, p_rk4_profe:float) -> np.ndarray:

        n_rk4_profe = np.zeros(len(p_rk4_profe))
        for i in range(len(p_rk4_profe)):

            n_rk4_profe[i] = ((tf_rk4_profe-t0_rk4_profe)/(p_rk4_profe[i])) +1 

        return n_rk4_profe
    
    n_rk4_profe = calculate_n_rk4_profe(tf_rk4_profe,t0_rk4_profe,p_rk4_profe)
    print('n', n_rk4_profe) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_rk4_profe(d_orificio_rk4_profe:float) -> float: 
        
        area_rk4_profe = (np.pi)*(d_orificio_rk4_profe/2)**2

        return area_rk4_profe
    area_rk4_profe = calculate_area_rk4_profe(d_orificio_rk4_profe)   
    print(area_rk4_profe)  

    def calculate_Q_rk4_profe(C_rk4_profe:float, area_rk4_profe:float, g_rk4_profe:float, h0_rk4_profe:float) -> float:
        
        Q_rk4_profe = C_rk4_profe*area_rk4_profe*np.sqrt(2*g_rk4_profe*h0_rk4_profe)

        return Q_rk4_profe
    
    Q_rk4_profe = calculate_Q_rk4_profe(C_rk4_profe, area_rk4_profe, g_rk4_profe, h0_rk4_profe)
    print(Q_rk4_profe)

    def calculate_V_rk4_profe(r_rk4_profe:float) -> float:

        V_rk4_profe = 4/3 * np.pi * r_rk4_profe**3

        return V_rk4_profe
    
    V_rk4_profe = calculate_V_rk4_profe(r_rk4_profe)
    print(V_rk4_profe)

    def calculate_t_estimativa_rk4_profe(V_rk4_profe:float, Q_rk4_profe:float) -> float:
    
        t_estimativa_rk4_profe = V_rk4_profe/Q_rk4_profe

        return t_estimativa_rk4_profe
    
    t_estimativa_rk4_profe = calculate_t_estimativa_rk4_profe(V_rk4_profe, Q_rk4_profe)
    print('t_estimativa', t_estimativa_rk4_profe)

    # Definindo a Função EDO:
    def f_rk4_profe(t_rk4_profe,h_rk4_profe):
        return (-C_rk4_profe * area_rk4_profe * np.sqrt(2*g_rk4_profe*h_rk4_profe))/((((2/3)*np.pi*h_rk4_profe)*(3*r_rk4_profe - h_rk4_profe)) - ((np.pi*h_rk4_profe**2)/3))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk4_profe = []
    h_array_rk4_profe = []

    # Método de rk4_profe
    for i in range(x_rk4_profe): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_rk4_profe = []
        t_list_rk4_profe = []
        m_rk4_profe = int(n_rk4_profe[i]) - 1 
        j_rk4_profe = 0
        h_rk4_profe = 2.75
        t_rk4_profe = 0
        while h_rk4_profe > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_rk4_profe = h_rk4_profe + p_rk4_profe[i]*f_rk4_profe(t_rk4_profe, h_rk4_profe)
                if h_rk4_profe <= 0:
                    break
                else:
                    t_rk4_profe = t_rk4_profe + p_rk4_profe[i]
                    t_list_rk4_profe.append(t_rk4_profe)
                    h_list_rk4_profe.append(h_rk4_profe)
        t_array_rk4_profe.append(t_list_rk4_profe)
        h_array_rk4_profe.append(h_list_rk4_profe)# guarda o vetor y dentro matriz y_array

    print('h', h_array_rk4_profe)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_rk4_profe)): # vai percorrer a matriz de y 

        t_rk4_profe = t_array_rk4_profe[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_rk4_profe = h_array_rk4_profe[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk4_profe, h_rk4_profe, linestyle='-', color=color, label='Passo ' + str(p_rk4_profe[i]))
    
    plt.title('Questão 2 - Solução Numérica por RK4-Profe')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk4_profe, "s"])
    tabela.add_row(["tf", tf_rk4_profe, "s"])
    tabela.add_row(["h0", h0_rk4_profe, "m"])
    tabela.add_row(["r", r_rk4_profe, "m"])
    tabela.add_row(["d_orifício", d_orificio_rk4_profe, "m"])
    tabela.add_row(["g", g_rk4_profe, "m/s²"])
    tabela.add_row(["C", C_rk4_profe, "-"])
    tabela.add_row(["p", p_rk4_profe, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_rk4_profe, "m³/s"])
    tabela.add_row(["V", V_rk4_profe, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_rk4_profe, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_rk4_profe)):
        t_rk4_profe = t_array_rk4_profe[i]
        h_rk4_profe = h_array_rk4_profe[i]
        for val1, val2 in zip(t_rk4_profe, h_rk4_profe):
            tabela.add_row([val1, val2])

    print(tabela)

    # Plotagem da Comparação das Soluções Numéricas:
    for i in range(len(t_array_euler_aula)):
        t_euler_aula = t_array_euler_aula[i]
        t_rk2_aula = t_array_rk2_aula[i]
        t_rk4_aula = t_array_rk4_aula[i]
        t_euler_profe = t_array_euler_profe[i]
        t_rk2_profe = t_array_rk2_profe[i]
        t_rk4_profe = t_array_rk4_profe[i]
        h_euler_aula = h_array_euler_aula[i]
        h_rk2_aula = h_array_rk2_aula[i]
        h_rk4_aula = h_array_rk4_aula[i]
        h_euler_profe = h_array_euler_profe[i]
        h_rk2_profe = h_array_rk2_profe[i]
        h_rk4_profe = h_array_rk4_profe[i]
        plt.plot(t_euler_aula, h_euler_aula, linestyle='-', color='#FF1493', label='Euler-Aula - Passo ' + str(p_rk4_profe[i]))
        plt.plot(t_euler_aula, h_rk2_aula, linestyle='-', color='#4B0082', label='RK2-Aula  - Passo ' + str(p_rk4_profe[i]))
        plt.plot(t_euler_aula, h_rk4_aula, linestyle='-', color='#006400', label='RK4-Aula  - Passo ' + str(p_rk4_profe[i]))
        plt.plot(t_euler_profe, h_euler_profe, linestyle='-', color='#FF1493', label='Euler-Profe  - Passo ' + str(p_rk4_profe[i]))
        plt.plot(t_euler_profe, h_rk2_profe, linestyle='-', color='#4B0082', label='RK2-Profe  - Passo ' + str(p_rk4_profe[i]))
        plt.plot(t_euler_profe, h_rk4_profe, linestyle='-', color='#006400', label='RK4-Profe  - Passo ' + str(p_rk4_profe[i]))
    plt.title('Comparação das Soluções Numéricas')
    plt.xlabel('Tempo(s)')
    plt.ylabel('h(t) [m]')
    plt.grid(True)
    plt.legend()
    plt.show()

comp = calculate_comparacao_modelos()