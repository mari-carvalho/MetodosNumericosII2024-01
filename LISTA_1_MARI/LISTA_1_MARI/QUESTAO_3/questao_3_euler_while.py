# Questão 3 - Solução Numérica - Método de Euler:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as Variáveis de Entrada:

g_euler = 9.81
R_euler = 6.37*10**6
x0_euler = 0 
v0_euler = 0 
vf_euler = 1500
h_euler = 0.25

# Definindo as funções de EDO do Sistema:

def fx_euler(t_euler, v_euler, x_euler):
    return v_euler

def fv_euler(t_euler, v_euler, x_euler):
    return (g_euler) * ((R_euler**2)/((R_euler + x_euler)**2))

# Método de Euler 
v_euler_list = []
x_euler_list = []
t_euler_list = []
v_euler = v0_euler
x_euler = x0_euler
t_euler = 0 
while v_euler <= vf_euler:
    x_euler = x_euler + h_euler*fx_euler(t_euler, v_euler, x_euler)
    v_euler = v_euler + h_euler*fv_euler(t_euler, v_euler, x_euler)
    if v_euler >= vf_euler:
        break 
    x_euler_list.append(x_euler)
    v_euler_list.append(v_euler)
    t_euler_list.append(t_euler)
    t_euler = t_euler + h_euler


print('x_euler', x_euler_list)
print('v_euler', v_euler_list)

# Plotando as soluções:

plt.plot(t_euler_list, v_euler_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('v(t)')
plt.grid()
plt.show()

plt.plot(t_euler_list, x_euler_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('x(t)')
plt.grid()
plt.show()

plt.plot(x_euler_list, v_euler_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.show()