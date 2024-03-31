# Questão 4 - Solução Numérica - Método de Euler e Runge-Kutta de 4ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Definindo as Variáveis de Entrada:

t0_euler = 0
tf_euler = 0.4
h_euler = 0.1
y0_euler = 2
z0_euler = 4
n_euler = ((tf_euler-t0_euler)/(h_euler)) + 1

# Definindo os Vetores:

t_euler = np.zeros(int(n_euler))
y_euler = np.zeros(int(n_euler))
z_euler = np.zeros(int(n_euler))

# Alimentando os vetores com as Condições Iniciais:

for i in range(int(n_euler)):
    if i == 0:
        t_euler[i] = t0_euler
        y_euler[i] = y0_euler
        z_euler[i] = z0_euler
    else: 
        t_euler[i] = i*h_euler
        y_euler[i] = 0
        z_euler[i] = 0

print('t_euler', t_euler)
print('y_euler', y_euler)
print('z_euler', z_euler)

# Definindo as funções do Sistema:

def fy_euler(t_euler, y_euler, z_euler):
    return (-2*y_euler) + (5*np.exp(-t_euler))

def fz_euler(t_euler, y_euler, z_euler):
    return (-y_euler*(z_euler**2))/(2)

# Método de Euler:

for i in range(0,int(n_euler) - 1):
    y_euler[i+1] = y_euler[i] + h_euler*fy_euler(t_euler[i], y_euler[i], z_euler[i])
    z_euler[i+1] = z_euler[i] + h_euler*fz_euler(t_euler[i], y_euler[i], z_euler[i])

print('y_euler', y_euler)
print('z_euler', z_euler)

# Plotando as soluções:

plt.plot(t_euler, y_euler, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('y(t)')
plt.grid()
plt.show()

plt.plot(t_euler, z_euler, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('z(t)')
plt.grid()
plt.show()


