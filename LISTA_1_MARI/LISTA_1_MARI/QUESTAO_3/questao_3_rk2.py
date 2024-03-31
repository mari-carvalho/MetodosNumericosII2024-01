# Questão 3 - Solução Numérica - Método de Runge-Kutta 2ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as Variáveis de Entrada:

g_rk2 = 9.81
R_rk2 = 6.37*10**6
vf_rk2 = 1500
v0_rk2 = 0
x0_rk2 = 0
t0_rk2 = 0
h_rk2 = 0.25

# Definindo as funções de EDO do Sistema:

def fx_rk2(t_rk2, v_rk2, x_rk2):
    return v_rk2

def fv_rk2(t_rk2, v_rk2, x_rk2):
    return (g_rk2) * ((R_rk2**2)/((R_rk2+x_rk2)**2))

# Método de Runge-Kutta 2ª Ordem:
v_rk2_list = []
x_rk2_list = []
t_rk2_list = []
v_rk2 = v0_rk2
x_rk2 = x0_rk2
t_rk2 = t0_rk2
while v_rk2 <= vf_rk2:
    k1_x_rk2 = h_rk2*fx_rk2(t_rk2, v_rk2, x_rk2)
    k2_x_rk2 = h_rk2*fx_rk2(t_rk2, v_rk2 + k1_x_rk2, x_rk2 + k1_x_rk2)
    x_rk2 = x_rk2 + (1/2)*(k1_x_rk2 + k2_x_rk2)
    k1_v_rk2 = h_rk2*fv_rk2(t_rk2, v_rk2, x_rk2)
    k2_v_rk2 = h_rk2*fv_rk2(t_rk2, v_rk2 + k1_v_rk2, x_rk2 + k1_v_rk2)
    v_rk2 = v_rk2 + (1/2)*(k1_v_rk2 + k2_v_rk2)
    if v_rk2 >= vf_rk2:
        break
    x_rk2_list.append(x_rk2)
    v_rk2_list.append(v_rk2)
    t_rk2_list.append(t_rk2)
    t_rk2 = t_rk2 + h_rk2

print('x_rk2_list', x_rk2_list)
print('v_rk2_list', v_rk2_list)

# Plotando as soluções:

plt.plot(t_rk2_list, v_rk2_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('v(t)')
plt.grid()
plt.show()

plt.plot(t_rk2_list, x_rk2_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('x(t)')
plt.grid()
plt.show()

plt.plot(x_rk2_list, v_rk2_list, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.show()