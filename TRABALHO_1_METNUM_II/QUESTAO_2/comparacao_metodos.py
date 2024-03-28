# Questão 2 - Comparação dos Métodos

# Resolução do Problema de PVI:
# Importando Bibliotecas =:
import numpy as np
import math as mt 
import matplotlib.pyplot as plt
from prettytable import PrettyTable 

def calculate_comparacao_metodos():

    # Método de Euler ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_euler = 9.81 # m/s²
    d_orificio_euler = 0.03 # m 
    r_euler = 1.5
    h0_euler = 2.75 # m 
    C_euler = 0.55 
    t0_euler = 0 
    tf_euler = 1000
    
    p_euler = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_euler = len(p_euler) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_euler(tf_euler:float, t0_euler: float, p_euler:float) -> np.ndarray:

        n_euler = np.zeros(len(p_euler))
        for i in range(len(p_euler)):

            n_euler[i] = ((tf_euler-t0_euler)/(p_euler[i])) +1 

        return n_euler
    
    n_euler = calculate_n_euler(tf_euler,t0_euler,p_euler)
    print('n', n_euler) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_euler(d_orificio_euler:float) -> float: 
        
        area_euler = (np.pi)*(d_orificio_euler/2)**2

        return area_euler
    area_euler = calculate_area_euler(d_orificio_euler)   
    print(area_euler)  

    def calculate_Q_euler(C_euler:float, area_euler:float, g_euler:float, h0_euler:float) -> float:
        
        Q_euler = C_euler*area_euler*np.sqrt(2*g_euler*h0_euler)

        return Q_euler
    
    Q_euler = calculate_Q_euler(C_euler, area_euler, g_euler, h0_euler)
    print(Q_euler)

    def calculate_V_euler(r_euler:float) -> float:

        V_euler = 4/3 * np.pi * r_euler**3

        return V_euler
    
    V_euler = calculate_V_euler(r_euler)
    print(V_euler)

    def calculate_t_estimativa_euler(V_euler:float, Q_euler:float) -> float:
    
        t_estimativa_euler = V_euler/Q_euler

        return t_estimativa_euler
    
    t_estimativa_euler = calculate_t_estimativa_euler(V_euler, Q_euler)
    print('t_estimativa', t_estimativa_euler)

    # Definindo a Função EDO:
    def f_euler(t_euler,h_euler):
        return (-C_euler * area_euler * np.sqrt(2*g_euler))/(np.sqrt(h_euler)*((2*np.pi*r_euler) - (np.pi*h_euler)))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_euler = []
    h_array_euler = []

    # Método de Euler
    for i in range(x_euler): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_euler = []
        t_list_euler = []
        m_euler = int(n_euler[i]) - 1 
        j_euler = 0
        h_euler = 2.75
        t_euler = 0
        while h_euler > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_euler = h_euler + p_euler[i]*f_euler(t_euler, h_euler)
                if h_euler <= 0:
                    break
                else:
                    t_euler = t_euler + p_euler[i]
                    t_list_euler.append(t_euler)
                    h_list_euler.append(h_euler)
        t_array_euler.append(t_list_euler)
        h_array_euler.append(h_list_euler)# guarda o vetor y dentro matriz y_array

    print('h', h_array_euler)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_euler)): # vai percorrer a matriz de y 

        t_euler = t_array_euler[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_euler = h_array_euler[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_euler, h_euler, linestyle='-', color=color, label='Solução Numérica- Passo ' + str(p_euler[i]))
    
    plt.title('Questão 2 - Método de Euler')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_euler, "s"])
    tabela.add_row(["tf", tf_euler, "s"])
    tabela.add_row(["h0", h0_euler, "m"])
    tabela.add_row(["r", r_euler, "m"])
    tabela.add_row(["d_orifício", d_orificio_euler, "m"])
    tabela.add_row(["g", g_euler, "m/s²"])
    tabela.add_row(["C", C_euler, "-"])
    tabela.add_row(["p", p_euler, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_euler, "m³/s"])
    tabela.add_row(["V", V_euler, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_euler, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_euler)):
        t_euler = t_array_euler[i]
        h_euler = h_array_euler[i]
        for val1, val2 in zip(t_euler, h_euler):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de RK2 ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_rk2 = 9.81 # m/s²
    d_orificio_rk2 = 0.03 # m 
    r_rk2 = 1.5
    h0_rk2 = 2.75 # m 
    C_rk2 = 0.55 
    t0_rk2 = 0 
    tf_rk2 = 1000
    
    p_rk2 = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk2 = len(p_rk2) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_rk2(tf_rk2:float, t0_rk2: float, p_rk2:float) -> np.ndarray:

        n_rk2 = np.zeros(len(p_rk2))
        for i in range(len(p_rk2)):

            n_rk2[i] = ((tf_rk2-t0_rk2)/(p_rk2[i])) +1 

        return n_rk2
    
    n_rk2 = calculate_n_rk2(tf_rk2,t0_rk2,p_rk2)
    print('n', n_rk2) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_rk2(d_orificio_rk2:float) -> float: 
        
        area_rk2 = (np.pi)*(d_orificio_rk2/2)**2

        return area_rk2
    area_rk2 = calculate_area_rk2(d_orificio_rk2)   
    print(area_rk2)  

    def calculate_Q_rk2(C_rk2:float, area_rk2:float, g_rk2:float, h0_rk2:float) -> float:
        
        Q_rk2 = C_rk2*area_rk2*np.sqrt(2*g_rk2*h0_rk2)

        return Q_rk2
    
    Q_rk2 = calculate_Q_rk2(C_rk2, area_rk2, g_rk2, h0_rk2)
    print(Q_rk2)

    def calculate_V_rk2(r_rk2:float) -> float:

        V_rk2 = 4/3 * np.pi * r_rk2**3

        return V_rk2
    
    V_rk2 = calculate_V_rk2(r_rk2)
    print(V_rk2)

    def calculate_t_estimativa_rk2(V_rk2:float, Q_rk2:float) -> float:
    
        t_estimativa_rk2 = V_rk2/Q_rk2

        return t_estimativa_rk2
    
    t_estimativa_rk2 = calculate_t_estimativa_rk2(V_rk2, Q_rk2)
    print('t_estimativa', t_estimativa_rk2)

    # Definindo a Função EDO:
    def f_rk2(t_rk2,h_rk2):
        return (-C_rk2 * area_rk2 * np.sqrt(2*g_rk2))/(np.sqrt(h_rk2)*((2*np.pi*r_rk2) - (np.pi*h_rk2)))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk2 = []
    h_array_rk2 = []

    # Método de rk2
    for i in range(x_rk2): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_rk2 = []
        t_list_rk2 = []
        m_rk2 = int(n_rk2[i]) - 1 
        j_rk2 = 0
        h_rk2 = 2.75
        t_rk2 = 0
        while h_rk2 > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_rk2 = h_rk2 + p_rk2[i]*f_rk2(t_rk2, h_rk2)
                if h_rk2 <= 0:
                    break
                else:
                    t_rk2 = t_rk2 + p_rk2[i]
                    t_list_rk2.append(t_rk2)
                    h_list_rk2.append(h_rk2)
        t_array_rk2.append(t_list_rk2)
        h_array_rk2.append(h_list_rk2)# guarda o vetor y dentro matriz y_array

    print('h', h_array_rk2)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_rk2)): # vai percorrer a matriz de y 

        t_rk2 = t_array_rk2[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_rk2 = h_array_rk2[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk2, h_rk2, linestyle='-', color=color, label='Passo ' + str(p_rk2[i]))
    
    plt.title('Questão 2 - Solução Numérica por RK2')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk2, "s"])
    tabela.add_row(["tf", tf_rk2, "s"])
    tabela.add_row(["h0", h0_rk2, "m"])
    tabela.add_row(["r", r_rk2, "m"])
    tabela.add_row(["d_orifício", d_orificio_rk2, "m"])
    tabela.add_row(["g", g_rk2, "m/s²"])
    tabela.add_row(["C", C_rk2, "-"])
    tabela.add_row(["p", p_rk2, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_rk2, "m³/s"])
    tabela.add_row(["V", V_rk2, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_rk2, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_rk2)):
        t_rk2 = t_array_rk2[i]
        h_rk2 = h_array_rk2[i]
        for val1, val2 in zip(t_rk2, h_rk2):
            tabela.add_row([val1, val2])

    print(tabela)

    # Método de rk4 ------------------------------------------------
    # Definindo Variáveis de Entrada:
    g_rk4 = 9.81 # m/s²
    d_orificio_rk4 = 0.03 # m 
    r_rk4 = 1.5
    h0_rk4 = 2.75 # m 
    C_rk4 = 0.55 
    t0_rk4 = 0 
    tf_rk4 = 1000
    
    p_rk4 = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x_rk4 = len(p_rk4) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n_rk4(tf_rk4:float, t0_rk4: float, p_rk4:float) -> np.ndarray:

        n_rk4 = np.zeros(len(p_rk4))
        for i in range(len(p_rk4)):

            n_rk4[i] = ((tf_rk4-t0_rk4)/(p_rk4[i])) +1 

        return n_rk4
    
    n_rk4 = calculate_n_rk4(tf_rk4,t0_rk4,p_rk4)
    print('n', n_rk4) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area_rk4(d_orificio_rk4:float) -> float: 
        
        area_rk4 = (np.pi)*(d_orificio_rk4/2)**2

        return area_rk4
    area_rk4 = calculate_area_rk4(d_orificio_rk4)   
    print(area_rk4)  

    def calculate_Q_rk4(C_rk4:float, area_rk4:float, g_rk4:float, h0_rk4:float) -> float:
        
        Q_rk4 = C_rk4*area_rk4*np.sqrt(2*g_rk4*h0_rk4)

        return Q_rk4
    
    Q_rk4 = calculate_Q_rk4(C_rk4, area_rk4, g_rk4, h0_rk4)
    print(Q_rk4)

    def calculate_V_rk4(r_rk4:float) -> float:

        V_rk4 = 4/3 * np.pi * r_rk4**3

        return V_rk4
    
    V_rk4 = calculate_V_rk4(r_rk4)
    print(V_rk4)

    def calculate_t_estimativa_rk4(V_rk4:float, Q_rk4:float) -> float:
    
        t_estimativa_rk4 = V_rk4/Q_rk4

        return t_estimativa_rk4
    
    t_estimativa_rk4 = calculate_t_estimativa_rk4(V_rk4, Q_rk4)
    print('t_estimativa', t_estimativa_rk4)

    # Definindo a Função EDO:
    def f_rk4(t_rk4,h_rk4):
        return (-C_rk4 * area_rk4 * np.sqrt(2*g_rk4))/(np.sqrt(h_rk4)*((2*np.pi*r_rk4) - (np.pi*h_rk4)))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array_rk4 = []
    h_array_rk4 = []

    # Método de rk4
    for i in range(x_rk4): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list_rk4 = []
        t_list_rk4 = []
        m_rk4 = int(n_rk4[i]) - 1 
        j_rk4 = 0
        h_rk4 = 2.75
        t_rk4 = 0
        while h_rk4 > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h_rk4 = h_rk4 + p_rk4[i]*f_rk4(t_rk4, h_rk4)
                if h_rk4 <= 0:
                    break
                else:
                    t_rk4 = t_rk4 + p_rk4[i]
                    t_list_rk4.append(t_rk4)
                    h_list_rk4.append(h_rk4)
        t_array_rk4.append(t_list_rk4)
        h_array_rk4.append(h_list_rk4)# guarda o vetor y dentro matriz y_array

    print('h', h_array_rk4)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array_rk4)): # vai percorrer a matriz de y 

        t_rk4 = t_array_rk4[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h_rk4 = h_array_rk4[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t_rk4, h_rk4, linestyle='-', color=color, label='Passo ' + str(p_rk4[i]))
    
    plt.title('Questão 2 - Solução Numérica por RK4')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0_rk4, "s"])
    tabela.add_row(["tf", tf_rk4, "s"])
    tabela.add_row(["h0", h0_rk4, "m"])
    tabela.add_row(["r", r_rk4, "m"])
    tabela.add_row(["d_orifício", d_orificio_rk4, "m"])
    tabela.add_row(["g", g_rk4, "m/s²"])
    tabela.add_row(["C", C_rk4, "-"])
    tabela.add_row(["p", p_rk4, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q_rk4, "m³/s"])
    tabela.add_row(["V", V_rk4, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa_rk4, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t)'])
    for i in range(len(t_array_rk4)):
        t_rk4 = t_array_rk4[i]
        h_rk4 = h_array_rk4[i]
        for val1, val2 in zip(t_rk4, h_rk4):
            tabela.add_row([val1, val2])

    print(tabela)

    # Plotagem da Comparação das Soluções Numéricas:
    for i in range(len(t_array_euler)):
        t_euler = t_array_euler[i]
        t_rk2 = t_array_rk2[i]
        t_rk4 = t_array_rk4[i]
        h_euler = h_array_euler[i]
        h_rk2 = h_array_rk2[i]
        h_rk4 = h_array_rk4[i]
        plt.plot(t_euler, h_euler, linestyle='-', color='#FF1493', label='Euler - Passo ' + str(p_euler[i]) )
        plt.plot(t_rk2, h_rk2, linestyle='-', color='#4B0082', label='RK2 - Passo ' + str(p_rk2[i]))
        plt.plot(t_rk4, h_rk4, linestyle='-', color='#006400', label='RK4 - Passo ' + str(p_rk4[i]))
    plt.title('Comparação das Soluções Numéricas')
    plt.xlabel('Tempo(s)')
    plt.ylabel('h(t) [m]')
    plt.grid(True)
    plt.legend()
    plt.show()


comp = calculate_comparacao_metodos()
