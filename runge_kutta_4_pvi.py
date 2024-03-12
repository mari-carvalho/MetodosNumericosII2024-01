# Resolução do Problema de PVI:
# Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

# Solução Numérica:

t0 = 0
tf = 4
y0 = 2
n = 5 #número de elementos
h = (tf-t0)/(n-1) # passo

def f1(t,y):
    return 4*np.exp(0.8*t) - 0.5*y

t = np.zeros(n)
y = np.zeros(n)

for i in range(0,n):
    if i == 0:
        t[i] = t0
        y[i] = y0
    else: 
        t[i] = i*h
        y[i] = 0

# Método de Runge-Kutta de 4ª Ordem:
        
for i in range(0,n-1):
    k1 = h*f1(t[i], y[i])
    k2 = h*f1(t[i] + (h/2), y[i] + (k1/2))
    k3 = h*f1(t[i] + (h/2), y[i] + (k2/2))
    k4 = h*f1(t[i] + h, y[i] + k3)
    y[i+1] = y[i] + (1/6)*(k1 + (2*k2) + (2*k3) + k4)
print(y[i+1])

yex = (4/1.3) * (np.exp(0.8*t) - np.exp(-0.5*t)) + 2*np.exp(-0.5*t)

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