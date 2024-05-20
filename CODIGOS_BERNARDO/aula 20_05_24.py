import numpy as np
import math
import matplotlib.pyplot as plt
import time

def Gauss_Seidel(A, b, x0, Eppara, maxit):
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)
    iter = 0
    Epest = np.linspace(100,100,ne)

    while np.max(Epest) >= Eppara and iter <= maxit:
        x_old = np.copy(x)

        for i in range(ne):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Critério de parada
        Epest = np.abs((x - x_old) / x) * 100

        iter += 1

    return x

def Gauss_Seidel_relax(A, b, x0, Eppara, maxit, Lambda):
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)

    iter = 0
    Epest = np.linspace(100,100,ne)

    while np.max(Epest) >= Eppara and iter <= maxit:
        x_old = np.copy(x)

        for i in range(ne):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Critério de parada
        Epest = np.abs((x - x_old) / x) * 100

        iter += 1

        # Relaxamento
        x = Lambda*x + (1-Lambda)*x_old

    return x

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
def TDMA(T_matriz, D):
    a = np.diagonal(T_matriz, offset = - 1)
    b = np.diagonal(T_matriz, offset=0)
    c = np.diagonal(T_matriz, offset=+1)
    d = D

    n = len(d)
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)
    x = np.zeros(n)

    # Forward elimination
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n-1):
        c_[i] = c[i] / (b[i] - a[i-1] * c_[i-1])
    for i in range(1, n):
        d_[i] = (d[i] - a[i-1] * d_[i-1]) / (b[i] - a[i-1] * c_[i-1])

    # Back substitution
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x

# Propriedades do Material - Cobre
rho = 8.92 # g/cm^3
cp = 0.092 # cal/(g.ºC)
k = 0.95 # cal/(cm.s.ºC)
L = 80 # c

# Dados Iniciais
tempo = 1000 # segundos
To = 20 # ºC
Tw = 0 # ºC
Te = 0 # ºC

# Malha
nx = 5
nt = 1000

# Cálculos Iniciais
alpha = k/(rho*cp)
dx = L/nx
dt = tempo/nt
rxt = alpha*dt/dx**2

# Critério de Scarvorought, 1966
n = 12 # Números de algarismos significativos
Eppara = 0.5*10**(2-n) # Termo relativos

# Número Máximo de Iterações
maxit = 1000

# Solução da Equação do Calor - MDFI
inicio = time.time()
# Inicialização das Matrizes
T_matriz = np.zeros((nx,nx))
T_old = np.ones(nx)*To
T = np.zeros((nt+1,nx))
D = np.zeros(nx)
h = 0
T[h,:] = To
t = 0

while t < tempo:
  #T_new = T_old

  h = h + 1
  for i in range(nx):
    # Condições de Contorno
    if i == 0:

      T_matriz[i,i] = 1 + 4*rxt
      T_matriz[i,i+1] = - 4/3*rxt

      D[i] = T_old[i] + 8/3*rxt*Tw

    elif i == nx-1:
      T_matriz[i,i-1] = - 4/3*rxt
      T_matriz[i,i] = 1 + 4*rxt

      D[i] = T_old[i] + 8/3*rxt*Te

    # Parte Central
    else:
      T_matriz[i,i-1] = - rxt
      T_matriz[i,i] = 1 + 2*rxt
      T_matriz[i,i+1] = - rxt

      D[i] = T_old[i]

  # Chute Inicial
  x0 = T_old
  #T_new = Gauss_Seidel(T_matriz, D, x0, Eppara, maxit)
  #T_new = Gauss_Seidel_relax(T_matriz, D, x0, Eppara, maxit, Lambda=1.0)
  #T_new = Jacobi(T_matriz, D, x0, Eppara, maxit)
  T_new = TDMA(T_matriz, D)

  T_old = T_new
  t = t + dt
  T[h, :] = T_new

fim = time.time()

# Calcula o tempo decorrido
tempo_decorrido = fim - inicio
print("Tempo decorrido:", tempo_decorrido, "segundos")

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