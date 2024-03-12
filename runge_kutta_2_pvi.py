# Resolução do Problema de PVI:
# Importando Bibliotecas =:
import numpy as np
import math as mt 
import matplotlib.pyplot as plt

# Solução Numérica:

t0 = 0
tf = 4
y0 = 2
n = 5
h = (tf-t0)/(n-1)

def f1(t,y):
    return 4*np.exp(0.8*t) - 0.5*y

t = np.zeros(n)
y = np.zeros(n)

for i in range(0,n):
    if i == 0:
        y[i] = y0
        t[i] = t0
    else:
        t[i] = i*h
        y[i] = 0

# Método de Runge-Kutta 2ª Ordem:

for i in range(0,n-1):
    k1 = h*f1(t[i], y[i])
    k2 = h*f1(t[i+1], y[i] + k1)
    y[i+1] = y[i] + (1/2)*(k1 + k2)
print(y[i+1])
yex = (4/1.3) * (np.exp(0.8*t) - np.exp(-0.5*t)) + 2*np.exp(-0.5*t)

plt.plot(t, y, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica')
plt.plot(t, yex, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata')
plt.xlabel('tempo (s)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

# Calculando o Erro Percentual Verdadeiro - Comparação de Analítica e Numérica:

erro = np.abs((yex-y)/yex)*100
print(erro)

plt.plot(t, erro, marker='o', linestyle='-', color='red', label='Erro Percentual Verdadeiro [%]')
plt.xlabel('tempo(s)')
plt.ylabel('Erro [%]')
plt.grid(True)
plt.legend()
plt.show()