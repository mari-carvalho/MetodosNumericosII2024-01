# Questão 4 - Solução Numérica - Método de Euler e Runge-Kutta de 4ª Ordem:

import numpy as np 
import matplotlib.pyplot as plt 

# Método de Euler:
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

# Método de Runge-Kutta de 4ª Ordem:
# Definindo as Variáveis de Entrada:

t0_rk4 = 0
tf_rk4 = 0.4
h_rk4 = 0.1
y0_rk4 = 2
z0_rk4 = 4
n_rk4 = ((tf_rk4-t0_rk4)/(h_rk4)) + 1

# Definindo os Vetores:

t_rk4 = np.zeros(int(n_rk4))
y_rk4 = np.zeros(int(n_rk4))
z_rk4 = np.zeros(int(n_rk4))

# Alimentando os vetores com as Condições Iniciais:

for j in range(int(n_rk4)):
    if j == 0 :
        print('j', j)
        t_rk4[j] = t0_rk4
        y_rk4[j] = y0_rk4
        z_rk4[j] = z0_rk4
    else:
        print('j', j)
        t_rk4[j] = j*h_rk4
        y_rk4[j] = 0
        z_rk4[j] = 0

print('t_rk4', t_rk4)
print('y_rk4', y_rk4)
print('z_rk4', z_rk4)

# Definindo as fuções do Sistema:

def fy_rk4(t_rk4, y_rk4, z_rk4):
    return (-2*y_rk4) + (5*np.exp(-t_rk4))

def fz_rk4(t_rk4, y_rk4, z_rk4):
    return (-y_rk4*(z_rk4**2))/(2)

# Método de Runge-Kutta 4ª Ordem:

for j in range(0,int(n_rk4) -1):
    k1_y_rk4 = h_rk4*fy_rk4(t_rk4[j], y_rk4[j], z_rk4[j])
    k2_y_rk4 = h_rk4*fy_rk4(t_rk4[j] + h_rk4/2, y_rk4[j] + k1_y_rk4/2, z_rk4[j] + k1_y_rk4/2)
    k3_y_rk4 = h_rk4*fy_rk4(t_rk4[j] + h_rk4/2, y_rk4[j] + k2_y_rk4/2, z_rk4[j] + k2_y_rk4/2)
    k4_y_rk4 = h_rk4*fy_rk4(t_rk4[j] + h_rk4, y_rk4[j] + k3_y_rk4, z_rk4[j] + k3_y_rk4)
    y_rk4[j+1] = y_rk4[j] + (1/6)*(k1_y_rk4 + 2*k2_y_rk4 + 2*k3_y_rk4 + k4_y_rk4)
    k1_z_rk4 = h_rk4*fz_rk4(t_rk4[j], y_rk4[j], z_rk4[j])
    k2_z_rk4 = h_rk4*fz_rk4(t_rk4[j] + h_rk4/2, y_rk4[j] + k1_z_rk4/2, z_rk4[j] + k1_z_rk4/2)
    k3_z_rk4 = h_rk4*fz_rk4(t_rk4[j] + h_rk4/2, y_rk4[j] + k2_z_rk4/2, z_rk4[j] + k2_z_rk4/2)
    k4_z_rk4 = h_rk4*fz_rk4(t_rk4[j] + h_rk4, y_rk4[j] + k3_z_rk4, z_rk4[j] + k3_z_rk4)
    z_rk4[j+1] = z_rk4[j] + (1/6)*(k1_z_rk4 + 2*k2_z_rk4 + 2*k3_z_rk4 + k4_z_rk4)

print('y_rk4', y_rk4)
print('z_rk4', z_rk4)

# Plotando as Soluções:

plt.plot(t_rk4, y_rk4, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('y(t)')
plt.grid()
plt.show()

plt.plot(t_rk4, z_rk4, linestyle='-', color='green', label='Solução Numérica')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('z(t)')
plt.grid()
plt.show()

# Comparando as soluções:

plt.plot(t_euler, y_euler, linestyle='-', color='green', label='Euler')
plt.plot(t_rk4, y_rk4, linestyle='-', color='blue', label='RK4/5')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('y(t)')
plt.grid()
plt.show()

plt.plot(t_euler, z_euler, linestyle='-', color='green', label='Euler')
plt.plot(t_rk4, z_rk4, linestyle='-', color='blue', label='RK4/5')

plt.legend()
plt.xlabel('Tempo(s)')
plt.ylabel('z(t)')
plt.grid()
plt.show()