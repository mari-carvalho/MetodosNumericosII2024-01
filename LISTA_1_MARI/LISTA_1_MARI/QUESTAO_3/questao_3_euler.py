# Questão 3 - Solução Numérica - Método de Euler:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as Variáveis de Entrada:

g_euler = 9.81
R_euler = 6.37*10**6
x0_euler = 0 
v0_euler = 0 
vf_euler = 1500
t0_euler = 0
tf_euler = 200
h_euler = 0.25
n_euler = ((tf_euler-t0_euler)/(h_euler)) + 1

# Criando os vetores:

v_euler = np.zeros(int(n_euler))
x_euler = np.zeros(int(n_euler))
t_euler = np.zeros(int(n_euler))

# Alimentando os Vetores com as Condições Iniciais:

for i in range(int(n_euler)):
    if i == 0:
        v_euler[i] = v0_euler
        x_euler[i] = x0_euler
        t_euler[i] = t0_euler
    elif i == int(n_euler) -1:
        v_euler[i] = vf_euler
        x_euler[i] = 0
        t_euler[i] = tf_euler
    else: 
        v_euler[i] = 0
        x_euler[i] = 0
        t_euler[i] = i*h_euler

print('v_euler', v_euler)
print('x_euler', x_euler)
print('t_euler', t_euler)

# Definindo as funções de EDO do Sistema:

def fx_euler(t_euler, v_euler, x_euler):
    return v_euler

def fv_euler(t_euler, v_euler, x_euler):
    return (g_euler) * ((R_euler**2)/((R_euler + x_euler)**2))

# Método de Euler 

for i in range(0,int(n_euler) - 1):
    x_euler[i+1] = x_euler[i] + h_euler*fx_euler(t_euler[i], v_euler[i], x_euler[i])
    v_euler[i+1] = v_euler[i] + h_euler*fv_euler(t_euler[i], v_euler[i], x_euler[i])

print('x_euler', x_euler)
print('v_euler', v_euler)

# Plotando as soluções:

plt.plot(t_euler, v_euler, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('v(t)')
plt.grid()
plt.show()

plt.plot(t_euler, x_euler, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('x(t)')
plt.grid()
plt.show()

plt.plot(x_euler, v_euler, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.grid()
plt.show()


