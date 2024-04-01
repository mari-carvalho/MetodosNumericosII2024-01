# Questão 4

import numpy as np
import matplotlib.pyplot as plt

# 1° passo, declarar as variáveis
#t[0, 0.4]
#y(0) = 2
#z(0) = 4

a = 0
b = 0.4
h = 0.1
n = int((b - a) / h) + 1

#Para y
def fyt(t, y):
    return -2 * y + 5 * np.exp(-t)

#Para z
def fyz(y, z):
    return -(y * z ** 2) / 2

# Euler
y_t = [0]
y_y = [2]
z_z = [4]

for i in range(n - 1):
    # Atualizando os "x"
    t_ynext = y_t[i] + h
    y_t.append(t_ynext)

    # Calculando Euler
    # para t
    y_ynext = y_y[i] + h * fyt(y_t[i], y_y[i])
    y_y.append(y_ynext)

    # para z
    z_znext = z_z[i] + h * fyz(y_y[i], z_z[i])
    z_z.append(z_znext)

# RK4
y_t2 = [0]
y_y2 = [2]
z_z2 = [4]

for i in range(n - 1):
    # Atualizando os "x"
    t_y2next = y_t2[i] + h
    y_t2.append(t_y2next)

    # Calculando RK4
    # para y
    k1y = h * fyt(y_t2[i], y_y2[i])
    k2y = h * fyt(y_t2[i] + h / 2, y_y2[i] + k1y / 2)
    k3y = h * fyt(y_t2[i] + h / 2, y_y2[i] + k2y / 2)
    k4y = h * fyt(y_t2[i] + h, y_y2[i] + k3y)
    y_y2next = y_y2[i] + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    y_y2.append(y_y2next)

    # para z
    k1z = h * fyz(y_y2[i], z_z2[i])
    k2z = h * fyz(y_y2[i] + h / 2, z_z2[i] + k1z / 2)
    k3z = h * fyz(y_y2[i] + h / 2, z_z2[i] + k2z / 2)
    k4z = h * fyz(y_y2[i] + h, z_z2[i] + k3z)
    z_z2next = z_z2[i] + (k1z + 2 * k2z + 2 * k3z + k4z) / 6
    z_z2.append(z_z2next)

# Crie uma figura com duas subplots lado a lado
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot para y
axs[0].set_title('Solução de $\\frac{dy}{dt} = -2y + 5e^{-t}$')
axs[0].plot(y_t, y_y, label='$y(t)$ - Euler')
axs[0].plot(y_t, y_y2, label='$y(t)$ - RK4')
axs[0].set_xlabel('$t$')
axs[0].set_ylabel('$y(t)$')
axs[0].legend()
axs[0].grid()

# Plot para z
axs[1].set_title('Solução de $\\frac{dz}{dt} = -\\frac{(yz^2)}{2}$')
axs[1].plot(y_t, z_z, label='$z(t)$ - Euler')
axs[1].plot(y_t, z_z2, label='$z(t)$ - RK4')
axs[1].set_xlabel('$t$')
axs[1].set_ylabel('$z(t)$')
axs[1].legend()
axs[1].grid()

# Ajuste o layout para evitar sobreposição
plt.tight_layout()

# Exiba o gráfico
plt.show()
