# Bibliotecas
import numpy as np
import math
import matplotlib.pyplot as plt
import time

ti = time.time()

def Jacobi(A, b, x0, Eppara, maxit):
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)

    iter = 0
    Epest = np.linspace(100,100,ne)

    while np.max(Epest) >= Eppara and iter <= maxit:
        x_old = np.copy(x)

        for i in range(ne):
            # Usar x_old para os termos antigos
            sum = np.dot(A[i, :i], x_old[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum) / A[i, i]

        # Critério de parada
        Epest = np.abs((x - x_old) / x) * 100

        iter += 1
   
    return x

#Propreidades do Material - Cobre:
rho = 8.92 # g/cm³
cp = 0.092 # cal/(g.ºC)
k = 0.95 # cal/(cm.s.ºC)
L = 80 # cm

# Dados Iniciais:
tempo = 1000 #s
T0 = 20 # ºC
Tw = 0 # ºC
Te = 0 # ºC

# Malha:
nx = 50
nt = 2000

# Calculos Iniciais:
alpha = k/(rho*cp) #difusividade térmica (m²/s)
dx = L/nx
dt=tempo/nt
rxt = alpha*(dt/(dx**2))

# Critério de Scarvorought, 1996
n = 12
Eppara = 0.5 * 10**(2 - n) # Termo Relatiovo

# Número Máximo de Iterações
maxit = 1000

# Solução da Equação do Calor - MDFI

#Inicialização das Matrizes
T_matriz = np.zeros((nx, nx))
T_old = np.ones(nx) * T0 # T_old vai sendo atualizado, por isso é um vetor de 1, porque se multiplicasse Temperatura por zeros não daria certo
T = np.zeros ((nt+1, nx)) # Matriz Solução
D = np.zeros(nx) # Resultado do produto matricial A * T 
h = 0 # Contador para iteração
T[h, :] = T0 # Contador para todas as colunas
t = 0 # Tempo

while t < tempo:
  h = h + 1
  T_matriz = np.zeros ((nx, nx))

  for i in range(nx):
    # Condições de Contorno:
    if i == 0: # Calculo do elemento a esquerda
      T_matriz[i, i] = 1 + 4 * rxt 
      T_matriz[i, i+1] = - 4/3 * rxt
      D[i] = T_old[i] + 8/3 * rxt * Te 

    elif i == nx-1: # Calculo do elemento a direita
      T_matriz[i, i-1] = - 4/3 * rxt
      T_matriz[i, i] = 1 + 4 * rxt
      D[i] = T_old[i] + 8/3 * rxt * Te

    else: # Calculo do elemento central
      T_matriz[i, i-1] = - rxt
      T_matriz[i, i] = 1 + 2 * rxt
      T_matriz[i, i+1] = - rxt
      D[i] = T_old[i]

  #Chute inicial
  x0 = T_old
  T_new = Jacobi(T_matriz, D, x0, Eppara, maxit)

  T_old = T_new # Para substituir o old para prosseguir calculando
  t = t + dt
  T[h, :] = T_new

    # Plotagem

X = np.linspace(0, L, nx)
Y = np.linspace(0, tempo, nt+1)
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, T, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
ax.set_xlabel('x (cm)')
ax.set_ylabel('t (segundos)')
ax.set_zlabel('T(x,t) (ºC)')
plt.title('Equação do Calor')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(30, 30)
plt.show()

tf = time.time()
time = tf - ti
print (time)