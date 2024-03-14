# Resolução do Problema de PVI:
#Importando Bibliotecas:
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# Definindo Variáveis de Entrada:
n = 150
tf = 400   
t0 = 0
T0 = 16
C0 = 1
h = (tf-t0)/(n-1)

# Definindo as Funções:
def f_C(t,T,C):
    return -np.exp(-10/(T+273)) * C

def f_T(t,T,C):
    return 1000 * np.exp(-10/(T+273)) * C - 10*(T-20)

# Definindo as Condições Iniciais:

t = np.zeros(n)
T = np.zeros(n)
C = np.zeros(n)

for i in range(0,n): # precisa de mais elementos que o numero de elementos, por causa do y[i+1]. Se for tempo de 0 a 4, 5 elementos, for precisa ir de 0 a 5 para dar 6 elementos 
    if i == 0:
        t[i] = t0
        T[i] = T0
        C[i] = C0
    else:
        t[i] = i*h 
        T[i] = 0
        C[i] = 0

# Método de Euler:

for i in range(0,n-1): # deve ir até n-1 porque a solução de y[i+1] é resolvida até o tempo de 3s, sendo que o tempo final é 4
    T[i+1] = T[i] + h*f_T(t[i], T[i], C[i])
    C[i+1] = C[i] + h*f_C(t[i], T[i], C[i])
print('T', T)
print('C', C)

#yex_v = np.sqrt((g*m)/cd) * np.tanh(np.sqrt((g*m)/cd)) FAREMOS A EXATA??
#yex_x = (m/cd) * (np.log(np.cosh(np.sqrt((g*cd)/m)*t)))

plt.plot(t, T, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica T')
#plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata X')
plt.xlabel('tempo(s)')
plt.ylabel('T(t)')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, C, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica C')
#plt.plot(t, yex_x, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata V')
plt.xlabel('tempo(s)')
plt.ylabel('C(t)')
plt.grid(True)
plt.legend()
plt.show() 

