# Resolução do Problema de PVI:
# Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

# Solução Numérica:

t0 = 0
tf = 2
y0 = 1
h = 0.5

def calculate_n(tf:float, t0:float, h:float) -> float:

    n = ((tf-t0)/h) + 1

    return n 

n = calculate_n(tf,t0,h)
print(n)

def f1(t,y):
    return (y*(t**3)) - 1.5*y

t = np.zeros(int(n))
y = np.zeros(int(n))

for i in range(0,int(n)):
    if i == 0:
        t[i] = t0
        y[i] = y0
    else: 
        t[i] = i*h
        y[i] = 0

# Método de Runge-Kutta de 4ª Ordem:
        
for i in range(0,int(n)-1):
    k1 = h*f1(t[i], y[i])
    k2 = h*f1(t[i] + (h/2), y[i] + (k1/2))
    k3 = h*f1(t[i] + (h/2), y[i] + (k2/2))
    k4 = h*f1(t[i] + h, y[i] + k3)
    y[i+1] = y[i] + (1/6)*(k1 + (2*k2) + (2*k3) + k4)
print('y', y[i+1])

yex = np.exp(((t**4)/4) - 1.5*t)

plt.plot(t, y, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica')
plt.plot(t, yex, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata')
plt.xlabel('tempo(s)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

# Calculando o Erro Percentual Verdaideiro - Comparação de Analítica e Numérica:

erro = np.abs((yex-y)/yex)*100
print(erro)

plt.plot(t, erro, marker='o', linestyle='-', color='red', label='Erro Percentual Verdadeiro [%]')
plt.xlabel('tempo(s)')
plt.ylabel('Erro [%]')
plt.grid(True)
plt.legend()
plt.show()