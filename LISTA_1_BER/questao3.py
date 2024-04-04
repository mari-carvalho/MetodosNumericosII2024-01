# Questão 3

import numpy as np
import matplotlib.pyplot as plt

# Constantes
g_0 = 9.81  # Aceleração gravitacional na superfície da Terra (m/s^2)
R = 6.37e6  # Raio da Terra (m)

# Função para a derivada de v em relação ao tempo (dv/dt = -g(0)*R^2 / (R + x)^2)
def dv_dt(x):
    return g_0 * R**2 / (R + x)**2

# Condições iniciais
vf = 1500
vi = 0
h = 0.1

# Listas para armazenar os resultados
v = [vi]
x = [0]
t = [0]
a = [0]

# Simulação usando o método de Euler com um loop while

i = 0 #i é a posição da lista

while v[i] < vf:
    # Calcula a próxima posição, velocidade e aceleração usando o método de Euler
    x_next = x[i] + h * v[i]
    v_next = v[i] + h * dv_dt(x[i])
    a_next = dv_dt(x_next)  # Agora, a aceleração é calculada usando a próxima posição
    # Adiciona os resultados às listas[]lskfnowenfqpeinqpefnpqiefn0
    x.append(x_next)
    v.append(v_next)
    a.append(a_next)

    # Atualiza o tempo
    t.append(t[i] + h)

    # Incrementa o contador
    i += 1

# Plotagem
plt.figure()
plt.plot(x, v, label='Euler')
plt.xlabel('Posição (m)')
plt.ylabel('Velocidade (m/s)')
plt.grid()
plt.legend()
plt.show()

x_rounded = round(x[-1], 1)
v_rounded = round(v[-1], 1)


print('A posição em que o foguete atingiu', v_rounded, 'm/s foi de:', x[-1], 'm')

# Listas para armazenar os resultados
v = [vi]
x = [0]
t = [0]

# Simulação usando o método de Euler modificado (método do ponto médio) com um loop while
i = 0
while v[i] < vf:
    # Método de Euler modificado
    k1 = h * dv_dt(x[i])
    k2 = h * dv_dt(x[i] + h/2)

    x_next = x[i] + h * v[i]
    v_next = v[i] + k2  # Atualização corrigida da velocidade

    # Adiciona os resultados às listas
    x.append(x_next)
    v.append(v_next)

    # Atualiza o tempo
    t.append(t[i] + h)

    # Incrementa o contador
    i += 1

# Plotagem
plt.figure()
plt.plot(x, v, label='RK23')
plt.xlabel('Posição (m)')
plt.ylabel('Velocidade (m/s)')
plt.grid()
plt.legend()
plt.show()

print('A posição em que o foguete atingiu', v_rounded, 'm/s foi de:', x[-1], 'm')


# Listas para armazenar os resultados
v = [vi]
x = [0]
t = [0]

i = 0
# Simulação usando o método de Runge-Kutta de quarta ordem (RK4) com um loop while
while v[i] < vf:
    # Método de Runge-Kutta de quarta ordem (RK4)
    k1 = h * dv_dt(x[i])
    k2 = h * dv_dt(x[i] + h/2)
    k3 = h * dv_dt(x[i] + h/2)
    k4 = h * dv_dt(x[i] + h)

    x_next = x[i] + h * v[i]
    v_next = v[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Adiciona os resultados às listas
    x.append(x_next)
    v.append(v_next)

    # Atualiza o tempo
    t.append(t[i] + h)

    # Incrementa o contador
    i += 1


# Plotagem
plt.figure()
plt.plot(x, v, label='RK23')
plt.xlabel('Posição (m)')
plt.ylabel('Velocidade (m/s)')
plt.grid()
plt.legend()
plt.show()
print('A posição em que o foguete atingiu', v_rounded, 'm/s foi de:', x[-1], 'm')
