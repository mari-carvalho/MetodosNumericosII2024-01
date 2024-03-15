# Resolução do Problema de PVI:
#Importando Bibliotecas:
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# Definindo Variáveis de Entrada:
g = 9.81 # m/s²
m = 68.1 # kg
cd = 0.25 # kg/m
n = 6 
tf = 10 
t0 = 0
y0_x = 0
y0_v = 0
h = (tf-t0)/(n-1)

# Definindo as Funções:
def f_x(v):
    return v

def f_v(v):
    return g - (cd/m)*v**2

# Definindo as Condições Iniciais:
t = np.zeros(n)
y_x = np.zeros(n)
y_v = np.zeros(n)

for i in range(0,n): # precisa de mais elementos que o numero de elementos, por causa do y[i+1]. Se for tempo de 0 a 4, 5 elementos, for precisa ir de 0 a 5 para dar 6 elementos 
    if i == 0:
        t[i] = t0
        y_x[i] = y0_x
        y_v[i] = y0_v
    else:
        t[i] = i*h 
        y_x[i] = 0
        y_v[i] = 0

# Método de Euler:

for i in range(0,n-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 3s, sendo que o tempo final é 4
    k1_x = h*f_x(y_v[i])
    k1_v = h*f_v(y_v[i])
    k2_x = h*f_x(y_x[i] + k1_x)
    k2_v = h*f_v(y_v[i] + k1_v)
    y_x[i+1] = y_x[i] + (1/2)*(k1_x + k2_x)
    y_v[i+1] = y_v[i] + (1/2)*(k1_v + k2_v)
print(y_x)
print(y_v)


yex_v = mt.sqrt((g*m)/cd) * np.tanh(mt.sqrt((g*m)/cd))
yex_x = (m/cd) * mt.log(10) * (np.cosh(mt.sqrt((g*cd)/m)*t))

plt.plot(t, y_x, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica X')
plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
plt.xlabel('tempo(s)')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, y_v, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica V')
plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
plt.xlabel('tempo(s)')
plt.ylabel('V(t)')
plt.grid(True)
plt.legend()
plt.show() 