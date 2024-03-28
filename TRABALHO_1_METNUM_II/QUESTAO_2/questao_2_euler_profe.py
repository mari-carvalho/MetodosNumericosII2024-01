# Questão 2 - Método de Euler - Professor

# Resolução do Problema de PVI:
# Importando Bibliotecas =:
import numpy as np
import math as mt 
import matplotlib.pyplot as plt
from prettytable import PrettyTable 

def calculate_2_euler():

    # Definindo Variáveis de Entrada:
    g = 9.81 # m/s²
    d_orificio = 0.03 # m 
    r = 1.5
    h0 = 2.75 # m 
    C = 0.55 
    t0 = 0 
    tf = 1000
    
    p = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
    x = len(p) # variável para medir o tamanho do vetor de passos h 

    # Definindo a Função que calcula o Número de Elementos:
    def calculate_n(tf:float, t0: float, p:float) -> np.ndarray:

        n = np.zeros(len(p))
        for i in range(len(p)):

            n[i] = ((tf-t0)/(p[i])) +1 

        return n
    
    n = calculate_n(tf,t0,p)
    print('n', n) # vetor com os números de elementos de cada passo solicitado pelo problema 

    def calculate_area(d_orificio:float) -> float: 
        
        area = (np.pi)*(d_orificio/2)**2

        return area
    area = calculate_area(d_orificio)   
    print(area)  

    def calculate_Q(C:float, area:float, g:float, h0:float) -> float:
        
        Q = C*area*np.sqrt(2*g*h0)

        return Q
    
    Q = calculate_Q(C, area, g, h0)
    print(Q)

    def calculate_V(r:float) -> float:

        V = 4/3 * np.pi * r**3

        return V
    
    V = calculate_V(r)
    print(V)

    def calculate_t_estimativa(V:float, Q:float) -> float:
    
        t_estimativa = V/Q

        return t_estimativa
    
    t_estimativa = calculate_t_estimativa(V, Q)
    print('t_estimativa', t_estimativa)

    # Definindo a Função EDO:
    def f(t,h):
        return (-C * area * np.sqrt(2*g*h))/((((2/3)*np.pi*h)*(3*r - h)) - ((np.pi*h**2)/3))
    
    # Definindo as Matrizes que vao guardar os Vetores de Tempo e de Y:
    t_array = []
    h_array = []

    # Método de Euler
    for i in range(x): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y. Vai percorrer o tamanho do vetor de passos, pois de acordo com o número de passos solicitados pelo problema, terá vetores de t e y 

        #t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        #h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        h_list = []
        t_list = []
        m = int(n[i]) - 1 
        j = 0
        h = 2.75
        t = 0
        while h > 0: # enquanto o valor da posição j no vetor h for maior que zero, o vetor será alimentado 
                h = h + p[i]*f(t, h)
                if h <= 0:
                    break
                t = t + p[i]
                t_list.append(t)
                h_list.append(h)
        t_array.append(t_list)
        h_array.append(h_list)# guarda o vetor y dentro matriz y_array

    print('h', h_array)

    # Plotagem da Solução Numérica:
    colors = ['#FF1493', '#4B0082']   # Pink, dark purple

    for i in range(len(h_array)): # vai percorrer a matriz de y 

        t = t_array[i] # vai adotar o vetor de t presente na matriz t_array correspondente a posição i 
        h = h_array[i] # vai adotar o vetor de y presente na matriz y_array correspondente a posição i 
        color = colors[i % len(colors)] # seleciona as cores pré definidas de cada linha 
        plt.plot(t, h, linestyle='-', color=color, label='Passo ' + str(p[i]))
    
    plt.title('Questão 2 - Solução Numérica RK2-Profe')
    plt.xlabel('Tempo (s)')
    plt.ylabel('h(t) [m]')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Criando Tabelas para guardar Parâmetros:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["t0", t0, "s"])
    tabela.add_row(["tf", tf, "s"])
    tabela.add_row(["h0", h0, "m"])
    tabela.add_row(["r", r, "m"])
    tabela.add_row(["d_orifício", d_orificio, "m"])
    tabela.add_row(["g", g, "m/s²"])
    tabela.add_row(["C", C, "-"])
    tabela.add_row(["p", p, "-"])
    print(tabela)

    # Criando Tabelas para guardar Parâmetros Calculados:
    tabela = PrettyTable()

    tabela.field_names = ['Parâmetros', 'Valores', 'Unidades']

    tabela.add_row(["Q", Q, "m³/s"])
    tabela.add_row(["V", V, "m³"])
    tabela.add_row(["t_estimativa", t_estimativa, "s"])
    print(tabela)

    tabela = PrettyTable(['t(s)', 'h(t) [m]'])
    for i in range(len(t_array)):
        t = t_array[i]
        h = h_array[i]
        for val1, val2 in zip(t, h):
            tabela.add_row([val1, val2])

    print(tabela)


euler2 = calculate_2_euler()
print(euler2)
