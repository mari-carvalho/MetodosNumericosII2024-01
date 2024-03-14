# Resolução do Problema de PVI usando o Método de Euler:
#Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

# Solução Numérica:

# Definindo Variáveis de Entrada:
t0 = 0
tf = 2
y0 = 1

h = [0.5, 0.25] # passos solicitados pelo problema # mais elementos, mais refinado, menor o erro 
x = len(h)

# Definindo as Funções:

# Função Passo:
def calculate_n(tf:float, t0: float, h:float) -> np.ndarray:

    n = np.zeros(len(h))
    for i in range(len(h)):
        n[i] = ((tf-t0)/(h[i])) +1 

    return n
n = calculate_n(tf,t0,h)
print(n) # número de elementos

#Função EDO:
def f(t,y):
    return y*t**3 - 1.5*y

# Definindo os Vetores de Tempo e Y:

t_array = []
y_array = []

for i in range(0,x): # colunas
    t = np.linspace(t0,tf,int(n[i]))
    y = np.zeros_like(t)
    y[0] = y0
    y_array.append(y)
    t_array.append(t)
print(t_array)
print(y_array)


for i in range(x): # loop para selecionar o vetor de t e o vetor de y que vão servir para os cálculos de y[i+1]
    t = t_array[i]
    y = y_array[i]
    print('t', t)
    print('y', y) 
    m = int(n[i]) - 1
    print('m', m)
    for j in range(0,m):
        y[j+1] = y[j] + h[i]*f(t[j], y[j])
    y_array[i] = y

print(y_array)

# Plotagem da Solução Numérica:

colors = ['#0000FF', '#1E90FF', '#4169E1', '#6495ED',   # Shades of blue
          '#87CEFA', '#ADD8E6', '#B0E0E6', '#87CEEB']   # More shades of blue

for i in range(len(y_array)):
    t = t_array[i]
    y = y_array[i]
    color = colors[i % len(colors)]
    plt.plot(t, y, marker='o', linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h[i]))
plt.xlabel('Tempo (s)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

# Solução Anlítica:

def calculate_yexata(t:np.ndarray) -> np.ndarray:

    y_exata_list = []

    for i in range(len(t)):
        
        y_exata = np.exp(((t[i]**4)/4) - 1.5*t[i])
        y_exata_list.append(y_exata)

    return y_exata_list

y_exata_array = []

for i in range(len(y_array)):
    t = t_array[i]
    print('t', t)
    y_exata = calculate_yexata(t)
    y_exata_array.append(y_exata)
print('y_exata_array', y_exata_array)

# Plotagem da Solução Analítica:

colors = ['#006400', '#008000', '#556B2F',   # Shades of dark green
          '#8FBC8F', '#98FB98', '#90EE90',   # Shades of light green
          '#00FF00', '#7CFC00', '#32CD32']   # Shades of green

for i in range(len(t_array)):
    t = t_array[i]
    print('t', t)
    y_exata = y_exata_array[i]
    print('y_exata', y_exata)
    color = colors[i % len(colors)]
    plt.plot(t, y_exata, marker='o', linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h[i]))
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

for i in range(len(t_array)):
    t = t_array[i]
    y = y_array[i]
    y_exata = y_exata_array[i]
    color = colors[i % len(colors)]
    plt.plot(t, y, marker='o', linestyle='-', color=color, label='Solução Numérica - Passo ' + str(h[i]))
    plt.plot(t, y_exata, marker='o', linestyle='-', color=color, label='Solução Analítica - Passo ' + str(h[i]))
plt.xlabel('Tempo(s)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

# Cálculo do Erro Percentual Verdadeiro:

def calculate_erro(y_exata:np.ndarray, y:np.ndarray) -> np.ndarray:

    erro_list = []
    for i in range(len(y_exata)):
        
        erro = np.abs((y_exata[i]-y[i])/y_exata[i])*100
        erro_list.append(erro)
    
    return erro_list

erro_array = []

for i in range(len(y_array)):
    y = y_array[i]
    y_exata = y_exata_array[i]
    erro = calculate_erro(y_exata, y)
    erro_array.append(erro)
print('erro', erro_array)

# Plotagem do Erro Percentual Verdadeiro vs Tempo:

colors = ['#FF4500', '#FF6347', '#FF7F50',   # Shades of dark orange
          '#FFA07A', '#FFDAB9', '#FFE4B5',   # Shades of light orange
          '#FF8C00', '#FFA500', '#FFD700']   # Shades of orange

for i in range(len(t_array)):
    t = t_array[i]
    erro = erro_array[i]
    color = colors[ i% len(colors)]
    plt.plot(t, erro, marker='o', linestyle='-', color=color, label='Erro Percentual Verdadeiro [%] - Passo ' + str(h[i]))
plt.xlabel('Tempo(s)')
plt.ylabel('Erro Percentual Verdadeiro [%]')
plt.grid(True)
plt.legend()
plt.show()