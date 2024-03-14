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
def f_x(t, x, v):
    return v

def f_v(t, x, v):
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
    k1_x = h*f_x(t[i], y_x[i],  y_v[i])
    k1_v = h*f_v(t[i], y_x[i], y_v[i])
    k2_x = h*f_x(t[i] + (h/2), y_x[i] + (k1_x/2),  y_v[i] + (k1_v/2))
    k2_v = h*f_v(t[i] + (h/2), y_x[i] + (k1_x/2), y_v[i] + (k1_v/2))
    k3_x = h*f_x(t[i] + (h/2), y_x[i] + (k2_x/2), y_v[i] + (k2_v/2))
    k3_v = h*f_v(t[i] + (h/2), y_x[i] + (k2_x/2), y_v[i] + (k2_v/2))
    k4_x = h*f_x(t[i] + h, y_x[i] + k3_x, y_v[i] + k3_v)
    k4_v = h*f_v(t[i] + h, y_x[i] + k3_x, y_v[i] + k3_v)
    y_x[i+1] = y_x[i] + (1/6)*(k1_x + (2*k2_x) + (2*k3_x) + k4_x)    
    y_v[i+1] = y_v[i] + (1/6)*(k1_v + (2*k2_v) + (2*k3_v) + k4_v)

print('x', y_x)
print('v', y_v)
print(t)

yex_v_list = [] 
yex_x_list = []

for i in range(len(t)):
    yex_v = np.sqrt((g*m)/cd) * np.tanh((np.sqrt((g*cd)/m))*t[i])
    yex_v_list.append(yex_v)
    yex_x = (m/cd) * (np.log(np.cosh(np.sqrt((g*cd)/m)*t[i])))
    yex_x_list.append(yex_x)

print('yex v', yex_v_list)
print('yex x', yex_x_list)


plt.plot(t, y_x, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica X')
plt.plot(t, yex_x_list, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
plt.xlabel('tempo(s)')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, y_v, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica V')
plt.plot(t, yex_v_list, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
plt.xlabel('tempo(s)')
plt.ylabel('V(t)')
plt.grid(True)
plt.legend()
plt.show() 